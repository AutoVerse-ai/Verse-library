function completed = example_nonlinearParam_reach_04_vanderPol_linearRemainder()
% example_nonlinearParam_reach_04_vanderPol_linearRemainder - example
%    of nonlinear reachability analysis with uncertain parameters; 
%
% Syntax:  
%    example_nonlinearParam_reach_04_vanderPol_linearRemainder
%
% Inputs:
%    no
%
% Outputs:
%    res - boolean 
%
% Example: 
%

% Author:       Victor Gassmann
% Written:      17-May-2019
% Last update:  23-April-2020 (restructure params/options)
% Last revision:---

%------------- BEGIN CODE --------------

% Parameters --------------------------------------------------------------

dim_x=2;
params.R0=zonotope([[1.0;1.0],[0.0 0;0 0]]);
params.U=zonotope([0,0]);
params.tFinal=4.0; %final time
params.paramInt=interval([-0.1;-0.1],[0.1;0.1]);

% Reachability Settings ---------------------------------------------------

options.timeStep=0.1; %time step size for reachable set computation
options.taylorTerms=5; %number of taylor terms for reachable sets
options.intermediateTerms = 4;
options.zonotopeOrder=10; %zonotope order
options.maxError = 1*ones(dim_x,1);
options.reductionInterval=1e3;
options.tensorOrder = 2;
options.verbose = false;
options.alg = 'lin';


% System Dynamics ---------------------------------------------------------

% vanderPol = nonlinearSys(@vanderPolEq);
simple2DParam = nonlinParamSys(@simple2DParamEq); %with uncertain parameters


% Reachability Analysis ---------------------------------------------------

tx1 = tic;
R_wo_linear = reach(simple2DParam, params, options); %with normal remainder
tComp1 = toc(tx1);
disp(['computation time of reachable set with normal lagrange remainder: ',num2str(tComp1)]);

tx2 = tic;
options.alg='linRem';
R_param = reach(simple2DParam, params, options); %remainder added to system matrices
tComp2 = toc(tx2);
disp(['computation time of reachable set with remainder added to system matrix: ',num2str(tComp2)]);


% Simulation --------------------------------------------------------------

simOpt.points = 60;
simOpt.fracVert = 0.0;
simOpt.fracInpVert = 0.0;
simOpt.inpChanges = 199;
% params = rmfield(params,'paramInt');

simRes = simulateRandom(simple2DParam, params, simOpt);


% Simulation --------------------------------------------------------------

projDims=[1 2];
plotOrder = 20;

figure; hold on; box on;

% reachable set: normal lagrange remainder
plotOverTime(R_wo_linear,2,'FaceColor',[.5 .5 .5]);

% reachable set: lagrange remainder added to system matrices (A,B)
plotOverTime(R_param,2,'FaceColor',[.7 .7 .7]);

% plot initial set
% plot(params.R0,projDims,'w','Filled',true,'EdgeColor','k');

% plot simulation results
for idx = 1:length(simRes)
    tmp = simRes(idx);
    plotOverTime(tmp,2,'k');
end
% label plot
xlabel('t');
ylabel('x2');


figure; hold on; box on;

% reachable set: normal lagrange remainder
plotOverTime(R_wo_linear,1,'FaceColor',[.5 .5 .5]);

% reachable set: lagrange remainder added to system matrices (A,B)
plotOverTime(R_param,1,'FaceColor',[.7 .7 .7]);

% plot initial set
% plot(params.R0,projDims,'w','Filled',true,'EdgeColor','k');

% plot simulation results
for idx = 1:length(simRes)
    tmp = simRes(idx);
    plotOverTime(tmp,1,'k');
end
% label plot
xlabel('t');
ylabel('x1');


% example completed
completed = 1;

%------------- END OF CODE --------------
