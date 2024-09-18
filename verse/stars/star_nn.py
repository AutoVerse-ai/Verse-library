import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from starset import *
from scipy.integrate import ode
from sklearn.decomposition import PCA
import pandas as pd

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

### synthetic dynamic and simulation function
def dynamic_test(vec, t):
    x, y = t # hack to access right variable, not sure how integrate, ode are supposed to work
    ### vanderpol
    # x_dot = y
    # y_dot = (1 - x**2) * y - x

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

    ###non-descript convergent system
    x_dot = y
    y_dot = -5*x-5*x**3-y
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

class PostNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PostNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc4(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.fc3(x)
        # x = self.relu(x)

        return x

C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
basis = np.array([[1, 0], [0, 1]]) * np.diag([.1, .1])
center = np.array([1.40,2.30])

input_size = 1    # Number of input features 
hidden_size = 64     # Number of neurons in the hidden layers -- this may change, I know NeuReach has this at default 64
# output_size = 1 + center.shape[0] # Number of output neurons -- this should stay 1 until nn outputs V instead of mu, whereupon it should reflect dimensions of starset
output_size = 1
# output_size = g.shape[0]

# if device is not None:
model = PostNN(input_size, hidden_size, output_size).to(device)

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Apply He Normal Initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Initialize biases to 0 (optional)

# Apply He initialization to the existing model
model.apply(he_init)

# Use SGD as the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

num_epochs: int = 50 # sample number of epoch -- can play with this/set this as a hyperparameter
num_samples: int = 100 # number of samples per time step
lamb: float = 5
batch_size: int = 5 # number of times computed at once, lower should be better but slower, min of 1

T = 14
ts = 0.2

initial_star = StarSet(center, basis, C, g)
# Toy Function to learn: x^2+20

times = torch.arange(0, T+ts, ts).to(device) # times to supply, right now this is fixed while S_t is random. can consider making this random as well

C = torch.tensor(C, dtype=torch.double).to(device)
g = torch.tensor(g, dtype=torch.float).to(device)
# Training loop
S_0 = sample_star(initial_star, num_samples*10) # should eventually be a hyperparameter as the second input, 
np.random.seed()

def sample_initial(num_samples: int = num_samples) -> List[List[float]]:
    samples = []
    for _ in range(num_samples):
        samples.append(S_0[np.random.randint(0, len(S_0))])
    return samples

def containment(points: torch.Tensor, times: torch.Tensor, bases: List[torch.Tensor], centers: List[torch.Tensor]):
    #  cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(bases[i])@(p-centers[i])-mu*g))
    #  the non-vectorized containment function for reference
    mu = model(times.unsqueeze(1))
    len_times = times.shape[0]
    dim = points.shape[2]

    shifted_points = points - torch.stack(centers).unsqueeze(0) 
    shifted_points_flat = shifted_points.view(num_samples * len_times, dim) # (n_samples*len_times, dim)
    bases_inv = torch.linalg.inv(torch.stack(bases)) # has shape (len_times, dim, dim)
    bases_inv_repeated = bases_inv.repeat(num_samples, 1, 1)  # Shape: (n_samples, n_times, point_dim, point_dim)
    # Reshape bases_inv_repeated to (n_samples * n_times, point_dim, point_dim)
    bases_inv_flat = bases_inv_repeated.view(num_samples * len_times, dim, dim)
    # Perform batched matrix multiplication
    transformed_points_flat = torch.bmm(bases_inv_flat, shifted_points_flat.unsqueeze(2)).squeeze(2)  # Shape: (n_samples * n_times, point_dim)
    transformed_points = transformed_points_flat.squeeze(1)  # Reshape back to (n_samples * len_times, point_dim)
    transformed_points = transformed_points.view(num_samples, len_times, dim)  # Reshape back to (n_samples, len_times, dim)
    # Apply C matrix (batch-matrix multiplication)
    transformed_points = torch.matmul(transformed_points, C.T)  # C has shape (k, dim), apply to all points
    # Apply ReLU and subtract time-dependent mu*g
    transformed_points = torch.relu(transformed_points - mu.view(1, len_times, 1) * g)
    # Compute vector norm for each point for each time step
    return torch.linalg.vector_norm(transformed_points, dim=2) 

for epoch in range(num_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()

    samples = sample_initial()

    post_points = []
    for point in samples:
            post_points.append(sim_test(None, point, T, ts).tolist())
    post_points = np.array(post_points) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]
    
    bases = [] # now grab V_t 
    centers = [] ### eventually this should be a NN output too
    # for i in range(len(times)):
    #     points = post_points[:, i, 1:]
    #     new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
    #     pca: PCA = PCA(n_components=points.shape[1])
    #     pca.fit(points)
    #     scale = np.sqrt(pca.explained_variance_)
    #     derived_basis = (pca.components_.T @ np.diag(scale)) # scaling each component by sqrt of dimension
    #     bases.append(torch.tensor(derived_basis))
    #     centers.append(torch.tensor(new_center))
    
    # ### V_t is now always I -- check that mu should go to zero
    for i in range(len(times)):
        points = post_points[:, i, 1:]
        new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
        bases.append(torch.eye(points.shape[1], dtype=torch.double).to(device))
        centers.append(torch.tensor(new_center).to(device))

    post_points = torch.tensor(post_points).to(device)

    for i in range(len(times)//batch_size+1):
        start = i*batch_size
        end = (i+1)*batch_size
        
        ### very naive way to do this, probably would want more or less equal batch sizes if not dividing equally

        mu = model(times[start:end].unsqueeze(1)) # get times in right form
        loss = torch.log1p(torch.sum(mu))+lamb*torch.sum(containment(post_points[:, start:end, 1:], times[start:end], bases[start:end], centers[start:end]))/num_samples
        loss.backward()
        optimizer.step()

    scheduler.step()
        # Print loss periodically
        # print(f'Loss: {loss.item():.4f}')
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}] \n_____________\n')
        print("Gradients of weights and loss", model.fc1.weight.grad, model.fc1.bias.grad)
        losses = 0
        for i in range(len(times)):
            t = torch.tensor([times[i]], dtype=torch.float32).to(device)
            mu = model(t)
            cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(bases[i])@(p-centers[i])-mu*g))
            loss = mu + lamb*torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))/len(post_points[:,i,1:])
            # loss = (1-lamb)*mu + lamb*torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))/len(post_points[:,i,1:])
            print(f'loss: {loss.item():.4f}, mu: {mu.item():.4f}, time: {t.item():.1f}')
            losses += loss.item()
        mu = model(times.unsqueeze(1)) # get times in right form
        other_loss = torch.sum(mu)+lamb*torch.sum(containment(post_points[:, :, 1:], times, bases, centers))/(num_samples)
        print(f'Losses: {losses:.4f}, ..., other loss {other_loss:.5f}')

# test the new model

model.eval()

### should probably use something to get local path but whatever
torch.save(model.state_dict(), "./verse/stars/model_weights.pth")

# S_0 = sample_star(initial_star, num_samples*10) ### this is critical step -- this needs to be recomputed per training step
S = sample_initial(num_samples*10)
post_points = []
for point in S:
        post_points.append(sim_test(None, point, T, ts).tolist())
post_points = np.array(post_points) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]

test_times = torch.arange(0, T+ts, ts)
test = torch.reshape(test_times, (len(test_times), 1))
bases = [] # now grab V_t 
centers = [] ### eventually this should be a NN output too
for i in range(len(times)):
    points = post_points[:, i, 1:]
    new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
    # pca: PCA = PCA(n_components=points.shape[1])
    # pca.fit(points)
    # scale = np.sqrt(pca.explained_variance_)
    # derived_basis = (pca.components_.T @ np.diag(scale)).T # scaling each component by sqrt of dimension
    # # plt.arrow(new_center[0], new_center[1], *derived_basis[0]
    # bases.append(torch.tensor(derived_basis))
    bases.append(torch.eye(points.shape[1], dtype=torch.double))
    centers.append(torch.tensor(new_center))


stars = []
percent_contained = []
cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(bases[i].T)@(p-centers[i])-model(test[i])*g))

model = model.to('cpu')
C, g = torch.Tensor.cpu(C), torch.Tensor.cpu(g)

for i in range(len(times)):
    # mu, center = model(test[i])[0].detach().numpy(), model(test[i])[1:].detach().numpy()
    stars.append(StarSet(centers[i], bases[i], C.numpy(), torch.relu(model(test[i])).detach().numpy()*g.numpy()))
    points = torch.tensor(post_points[:, i, 1:])
    contain = torch.sum(torch.stack([cont(point, i) == 0 for point in points]))
    percent_contained.append(contain/(num_samples*10)*100)
    # stars.append(StarSet(center, bases[i], C.numpy(), mu*g.numpy()))
    # stars.append(StarSet(centers[i], bases[i], C.numpy(), np.diag(model(test[i]).detach().numpy())@g.numpy()))

percent_contained = np.array(percent_contained)
# for t in test:
#      print(model(t), t)
# for b in bases:
#      print(b)
# plt.plot(test_times, model(test).detach().numpy())
plot_stars_points_nonit(stars, post_points)

results = pd.DataFrame({
    'time': test.squeeze().numpy(),
    'mu': model(test).squeeze().detach().numpy(),
    'percent of points contained': percent_contained
})

results.to_csv('./verse/stars/nn_results.csv', index=False)
plt.show()