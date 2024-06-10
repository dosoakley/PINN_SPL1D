%Try making a PINN for the 1D stream power law and compare it to a finite difference solution.

use_lbfgs = false; %Use LBFGS solver for PINN. If false, Adam is used.

%Set the stream power law constants K, n, and m.
%These are set to the default ttlem values, which were what we used in
%MakeUpliftOnlyModel.m.
m = 0.5;
n = 1;
K = 3e-6;

%Create the regular grids in x and t that will be used for the initial and
%boundary conditions.
dx = 50;
xL = 10e3;
% xL = 20e3;
x_grid = 0:dx:xL; %A 10 km long stream, with 50 m cells.
dt = 1e3;
total_time = 2e6;
t_grid = 0:dt:total_time;

%Create initial condition points on a regular grid.
%Initial conditions are that it is flat everywhere.
x0 = x_grid;
t0 = zeros(size(x0));
% z0 = zeros(size(x0));
z0 = 1e-4*x0; %Give it a very slight slope instead.

%Create the boundary conditions, also on a regular grid.
%On the left BC, the boundary condition is that z = 0.
% %On the right BC, the boundary condition is that z = (1e-3)*t, where 1e-3 is the uplift rate.
% xBC = [zeros(1,length(t_grid)),xL*ones(1,length(t_grid))];
% tBC = [t_grid,t_grid];
% zBC = [zeros(1,length(t_grid)),(1e-3)*t_grid];
%I've changed this to only have the left BC.
xBC = [zeros(1,length(t_grid))];
tBC = [t_grid];
zBC = [zeros(1,length(t_grid))];

%Create the internal collocation points, including their uplift rates.
%This is a model consisting of block uplift, with the block starting at 2
%km from the stream outlet.
numInternalCollocationPoints = 5000;
pointSet = sobolset(2); %I could just do random points, but this will give a more even filling of the space.
points = net(pointSet,numInternalCollocationPoints);
x = points(:,1)*xL;
t = points(:,2)*total_time;
u = zeros(size(x));
u(x > 2e3) = 1e-3;

%Build up the neural network.
numNeurons = 50; %Number of neurons per layer.
numHidden = 5; %Number of layers (excluding the input and output layers).
layers = featureInputLayer(2,Name="Input"); %Inputs: x and t.
for i = 1:numHidden
    layers = [layers,fullyConnectedLayer(numNeurons,Name="fc"+i),tanhLayer(Name="tanh"+i)];
end
layers = [layers,fullyConnectedLayer(1,Name="Output")]; %Outputs: Ground elevation (z) and water depth (d).
PINN = dlnetwork(layers);
PINN = dlupdate(@double,PINN); %Convert it to use type double. This should increase accuracy, but may also be slower and not work on GPU. (Optional.)

%Convert the training data to dlarray objects.
%I don't completely understand the formats yet.
%BC seems to give me a 1 x N dlarray, and CB an N x 1 dlarray.
x = dlarray(x,"BC");
t = dlarray(t,"BC");
u = dlarray(u,"BC");
x0 = dlarray(x0',"BC");
t0 = dlarray(t0',"BC");
z0 = dlarray(z0',"BC");
xBC = dlarray(xBC',"BC");
tBC = dlarray(tBC',"BC");
zBC = dlarray(zBC',"BC");

%Initialize the parameters for the Adam solver.
averageGrad = [];
averageSqGrad = [];
learning_rate = 1e-3;
max_iterations = 2e4;

%Accelerate the loss function.
accfun = dlaccelerate(@modelLoss);

%Create LBFGS loss function.
lossFcnLBFGS = @(net) dlfeval(accfun,net,x,t,u,x0,t0,z0,xBC,tBC,zBC,xL,total_time,K,n,m);

%Specify LBFGS optimization options.
gradientTolerance = 1e-5;
stepTolerance = 1e-5;
solverState = lbfgsState;

%Train the PINN.
all_loss = zeros(max_iterations,1);
iteration = 0;
start = tic;
while iteration < max_iterations
    iteration = iteration + 1;
    
    if use_lbfgs

        %Update the network parameters using lbfgs.
        [PINN, solverState] = lbfgsupdate(PINN,lossFcnLBFGS,solverState);
        loss = solverState.Loss;

        %Stop early if LBFGS tolerances reached or LBFGS failed.
        if solverState.GradientsNorm < gradientTolerance || ...
                solverState.StepNorm < stepTolerance || ...
                solverState.LineSearchStatus == "failed"
            break
        end

    else

        % Evaluate the model loss and gradients using dlfeval and the
        % modelLoss function.
        [loss,gradients] = dlfeval(accfun,PINN,x,t,u,x0,t0,z0,xBC,tBC,zBC,xL,total_time,K,n,m);
    
        % Update the network parameters using the adamupdate function.
        [PINN,averageGrad,averageSqGrad] = adamupdate(PINN,gradients,averageGrad, ...
            averageSqGrad,iteration,learning_rate);
    end

    %Keep track of the training progress.
    all_loss(iteration) = double(gather(extractdata(loss)));

    %Print out a summary.
    if mod(iteration,100) == 0
        D = duration(0,0,toc(start),Format="hh:mm:ss");
        disp("Iteration: " + iteration + ", Elapsed: " + string(D) + ", Loss: " + all_loss(iteration))
    end
end

%Plot the training progress.
figure(1)
% plot(1:iteration,all_loss(1:iteration))
semilogy(1:iteration,all_loss(1:iteration))
xlabel("Iteration")
ylabel("Loss")
grid on

%Evaluate the model on the grid at the final time, and plot it.
xf = dlarray(x_grid',"BC");
tf = dlarray(total_time*ones(length(x_grid),1),"BC");
zf = EvaluatePINN(PINN,xf,tf,xL,total_time);


%Run the finite difference (FD) model and compare it to the PINN model.
nsteps = round(total_time/dt);
z_grid = extractdata(z0);
A_grid = GetArea(x_grid,xL);
u_grid = zeros(size(x_grid));
u_grid(x_grid > 2e3) = 1e-3;
for istep = 1:nsteps
    z_grid = RunStepForward(x_grid,z_grid,A_grid,u_grid,m,n,K,dt);
end

%Plot the PINN and FD models to compare.
figure(2)
plot(x_grid/1e3,z_grid,xf/1e3,zf);
xlabel('Distance Along Stream (km)')
ylabel('Elevation(m)')
legend('Finite Difference','PINN')

function A = GetArea(x,xL)
%Estimate drainage areas using Hack's law.
% The numbers, and the idea of estimating area this way, come from this example: 
% https://fastscapelib.readthedocs.io/en/latest/examples/river_profile_py.html
hack_coef = 6.69;
hack_exp = 1.67;
downstream_dist = xL-x; %x = 0 at the outlet, so this just reverses it to start at the head.
A = hack_coef * downstream_dist.^hack_exp;
end

%Define a function for evaluating z at a point with appropriate scaling.
function z = EvaluatePINN(net,x,t,dist_scale,time_scale)
%Evaluate at the given points.
xt = cat(1,x./dist_scale+eps,t./time_scale+eps); %Scale and center the variables for input into the neural network. This will put them in the range [-1,1].
z = forward(net,xt) * 1e3;
end

%Define the model loss function.
function [loss,gradients] = modelLoss(net,x,t,u,x0,t0,z0,xBC,tBC,zBC,xL,tmax,K,n,m)

%Calculate the forward model at the internal points, and compute its gradients.
z = EvaluatePINN(net,x,t,xL,tmax);
gradz = dlgradient(sum(z,"all"),{x,t},EnableHigherDerivatives=true); 
S = gradz{1}; %Slope
dzdt = gradz{2}; %Rate of change of elevation.

%Calculate the erosion rate loss term, based on the difference between the
%model and the stream power law.
A = GetArea(x,xL);
er = K .* A.^m .* S.^n; %Erosion rate according to the stream power law.
dzdt_target = u - er; %The expected rate of change in elevation is the uplift rate minus the erosion rate.
er_loss = l2loss(dzdt,dzdt_target);

%Calculate the loss for the initial conditions.
z0_pred = EvaluatePINN(net,x0,t0,xL,tmax);
IC_loss = l2loss(z0_pred,z0);

%Calculate the loss for the boundary conditions.
zBC_pred = EvaluatePINN(net,xBC,tBC,xL,tmax);
BC_loss = l2loss(zBC_pred,zBC);

%Combine all the loss functions together.
% Calculated loss to be minimized by combining errors.
% loss = er_loss + IC_loss + BC_loss;
loss = er_loss*1e6 + IC_loss/1e6 + BC_loss/1e6; %Scale these so they are at least sort of in the same range.

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,net.Learnables);

end

function z = RunStepForward(x,z,A,u,m,n,K,dt)
%Run a single fastscape step for n = 1.
%This works only for a single stream, specified as a list of distance (x) and
%elevation (z) coordinates, starting from the stream outlet, which is
%assumed to be at a fixed elevation.
%The first node of the stream has a 0 erosion rate BC.
if n == 1
    z = z + u*dt; % add uplift
    for i = 2:numel(x)
        tt = K*A(i)^m*dt/(x(i)-x(i-1));
        z(i) = (z(i) + z(i-1)*tt)/(1+tt); %Eqn. 22 of Braun and Willett (2013).
    end
elseif n ~= 1
    error('n~=1 is not implemented yet.')
end
end