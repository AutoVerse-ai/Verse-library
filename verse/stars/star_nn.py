import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from starset import *
from scipy.integrate import ode
from sklearn.decomposition import PCA

### synthetic dynamic and simulation function
def dynamic_test(vec, t):
    x, y = t # hack to access right variable, not sure how integrate, ode are supposed to work
    ### vanderpol
    x_dot = y
    y_dot = (1 - x**2) * y - x

    ### cardiac cell
    # x_dot = -0.9*x*x-x*x*x-0.9*x-y+1
    # y_dot = x-2*y

    ### jet engine
    # x_dot = -y-1.5*x*x-0.5*x*x*x-0.5
    # y_dot = 3*x-y

    ### brusselator 
    # x_dot = 1+x**2*y-2.5*x
    # y_dot = 1.5*x-x**2*y

    ### bucking col -- change center to around -0.5 and keep basis size low
    # x_dot = y
    # y_dot = 2*x-x*x*x-0.2*y+0.1
    return [x_dot, y_dot]

def sim_test(
    mode: List[str], initialCondition, time_bound, time_step, 
) -> np.ndarray:
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    # note: digit of time
    init = list(initialCondition)
    trace = [[0] + init]
    for i in range(len(t)):
        r = ode(dynamic_test)
        r.set_initial_value(init)
        res: np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init)
    return np.array(trace)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        
        return x

input_size = 1     # Number of input features -- this should change to reflect dimensions of starset 
hidden_size = 64     # Number of neurons in the hidden layers -- this may change, I know NeuReach has this at default 64
output_size = 1     # Number of output neurons -- this should stay 1 until nn outputs V instead of mu, whereupon it should reflect dimensions of starset

model = SimpleNN(input_size, hidden_size, output_size)

# Use SGD as the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

num_epochs = 50 # sample number of epoch -- can play with this/set this as a hyperparameter
num_samples = 50 # number of samples per time step

T = 7
ts = 0.1

C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
basis = np.array([[1, 0], [0, 1]]) * np.diag([0.01, 0.01])
center = np.array([1.40,2.30])
initial_star = StarSet(center, basis, C, g)
# Toy Function to learn: x^2+20
to_learn = lambda x : x**2+20

inputs = torch.randn(50, 1)  # 50 inputes to lear
labels = inputs * inputs + 20 # apply_ is inplace which I don't want, just use torch multiplication

times = np.arange(0, T+ts, ts) # times to supply, right now this is fixed while S_t is random. can consider making this random as well

C = torch.tensor(C, dtype=torch.double)
g = torch.tensor(g, dtype=torch.double)
lam = 100
# Training loop
for epoch in range(num_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()

    S_0 = sample_star(initial_star, num_samples) ### this is critical step -- this needs to be recomputed per training step
    post_points = []
    for point in S_0:
            post_points.append(sim_test(None, point, T, ts).tolist())
    post_points = np.array(post_points) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]
    
    bases = [] # now grab V_t 
    centers = [] ### eventually this should be a NN output too
    for i in range(len(times)):
        points = post_points[:, i, 1:]
        new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
        pca: PCA = PCA(n_components=points.shape[1])
        pca.fit(points)
        scale = np.sqrt(pca.explained_variance_)
        derived_basis = (pca.components_.T @ np.diag(scale)).T # scaling each component by sqrt of dimension
        
        bases.append(torch.tensor(derived_basis))
        centers.append(torch.tensor(new_center))

    post_points = torch.tensor(post_points)
    ### for now, don't worry about batch training, just do single input, makes more sense to me to think of loss function like this
    ### I would really like to be able to do batch training though, figure out a way to make it work
    for i in range(len(times)):
        # Forward pass
        t = torch.tensor([times[i]], dtype=torch.float32)
        mu = model(t)
        
        # Compute the loss
        cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(bases[i])@(p-centers[i])-mu*g))
        loss = mu + torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))

        # Backward pass and optimize
        # pretty sure I'll need to modify this if I'm not doing batch training 
        # will just putting optimizer on the earlier for loop help?
        loss.backward()
        if i==50:
            print(model.fc1.weight.grad, model.fc1.bias.grad)
        optimizer.step()
        
        print(f'Loss: {loss.item()}, mu: {mu.item()}, t: {t}')

    scheduler.step()
    # Print loss periodically
    # print(f'Loss: {loss.item():.4f}')
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# test the new model
# test_inputs = torch.randn(25, 1)
# test_lables = test_inputs*test_inputs + 20

model.eval()

test_times = torch.arange(0, T, ts)
test = torch.reshape(test_times, (len(test_times), 1))
print(model(test), test)