import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from starset import *
from scipy.integrate import ode
from sklearn.decomposition import PCA
import pandas as pd

class PostNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PostNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.fc3(x)
        x = self.relu(x)

        return x

C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
basis = np.array([[1, 0], [0, 1]]) * np.diag([.1, .1])
center = np.array([1.40,2.30])

def create_model(input_size: int = 1, hidden_size: int = 64, output_size: int =1) -> PostNN:
    return PostNN(input_size, hidden_size, output_size)

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Apply He Normal Initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Initialize biases to 0 (optional)

def model_he_init(model: PostNN) -> None: 
    def he_init(m) -> None: 
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Apply He Normal Initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases to 0 (optional)
    
    model.apply(he_init)
    
def train(initial: StarSet, sim: Callable, model: PostNN, mode_label: int = None, num_epochs: int = 50, num_samples: int = 100, T: float = 7, ts: float=0.1, lamb: float = 7) -> None:
    # Use SGD as the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    times = torch.arange(0, T+ts, ts) # times to supply, right now this is fixed while S_t is random. can consider making this random as well

    C = torch.tensor(initial.C, dtype=torch.double)
    g = torch.tensor(initial.g, dtype=torch.float)
    # Training loop
    S_0 = sample_star(initial, num_samples*10) # should eventually be a hyperparameter as the second input, 
    np.random.seed()

    def sample_initial(num_samples: int = num_samples) -> List[List[float]]:
        samples = []
        for _ in range(num_samples):
            samples.append(S_0[np.random.randint(0, len(S_0))])
        return samples

    for epoch in range(num_epochs):
        # Zero the parameter gradients
        optimizer.zero_grad()

        samples = sample_initial()

        post_points = []
        for point in samples:
                post_points.append(sim(mode_label, point, T, ts).tolist())
        post_points = np.array(post_points) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]
        
        bases = [] # now grab V_t 
        centers = [] ### eventually this should be a NN output too
        for i in range(len(times)):
            points = post_points[:, i, 1:]
            new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
            pca: PCA = PCA(n_components=points.shape[1])
            pca.fit(points)
            scale = np.sqrt(pca.explained_variance_)
            derived_basis = (pca.components_.T @ np.diag(scale)) # scaling each component by sqrt of dimension
            bases.append(torch.tensor(derived_basis))
            centers.append(torch.tensor(new_center))

        post_points = torch.tensor(post_points)
        for i in range(len(times)):
            # Forward pass
            t = torch.tensor([times[i]], dtype=torch.float32)
            mu = model(t)

            # Compute the loss
            cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(bases[i]+ 1e-6*torch.eye(n))@(p-centers[i])-mu*g))
            loss = torch.log1p(mu) + lamb*torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))/num_samples 

            loss.backward()
            optimizer.step()
            
        scheduler.step()

        # # Print loss periodically
        # # print(f'Loss: {loss.item():.4f}')
        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}] \n_____________\n')
        #     # print("Gradients of weights and loss", model.fc1.weight.grad, model.fc1.bias.grad)
        #     for i in range(len(times)):
        #         t = torch.tensor([times[i]], dtype=torch.float32)
        #         mu = model(t)
        #         cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(bases[i])@(p-centers[i])-mu*g))
        #         loss = torch.log1p(mu)  + lamb*torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))/len(post_points[:,i,1:])
        #         # loss = (1-lamb)*mu + lamb*torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))/len(post_points[:,i,1:])
        #         print(f'loss: {loss.item():.4f}, mu: {mu.item():.4f}, time: {t.item():.1f}')

def get_model(initial: StarSet, sim: Callable, mode_label: int = None, num_epochs: int = 50, num_samples: int = 100, T: float = 7, ts: float=0.1, lamb: float = 7, input_size: int = 1, hidden_size: int = 64, output_size: int = 1) -> PostNN:
    model = create_model(input_size, hidden_size, output_size)
    model_he_init(model)
    train(initial, sim, model, mode_label, num_epochs, num_samples, T, ts, lamb)
    return model

def gen_reachtube(initial: StarSet, sim: Callable, model: PostNN, mode_label: int = None, num_samples: int=1000, T: float = 7, ts: float = 0.1) -> List[StarSet]:
    S = sample_star(initial, num_samples)
    post_points = []
    for point in S:
            post_points.append(sim(mode_label, point, T, ts).tolist())
    post_points = np.array(post_points) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]

    test_times = torch.arange(0, T+ts, ts)
    test = torch.reshape(test_times, (len(test_times), 1))
    bases = [] # now grab V_t 
    centers = [] ### eventually this should be a NN output too
    for i in range(len(test_times)):
        points = post_points[:, i, 1:]
        new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
        pca: PCA = PCA(n_components=points.shape[1])
        pca.fit(points)
        scale = np.sqrt(pca.explained_variance_)
        # print(pca.components_, scale)
        derived_basis = (pca.components_.T @ np.diag(scale)).T # scaling each component by sqrt of dimension
        # derived_basis = (pca.components_.T ).T # scaling each component by sqrt of dimension
        # print(pca.components_[0]*scale[0], pca.components_[1]*scale[1])
        # plt.arrow(new_center[0], new_center[1], *derived_basis[0]
        bases.append(torch.tensor(derived_basis))
        # bases.append(torch.eye(points.shape[1], dtype=torch.double))
        centers.append(torch.tensor(new_center))
    
    C = torch.tensor(initial.C, dtype=torch.double)
    g = torch.tensor(initial.g, dtype=torch.float)

    stars = []
    cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(bases[i].T)@(p-centers[i])-model(test[i])*g))
    for i in range(len(test_times)):
        # mu, center = model(test[i])[0].detach().numpy(), model(test[i])[1:].detach().numpy()
        stars.append(StarSet(centers[i], bases[i], C.numpy(), torch.relu(model(test[i])).detach().numpy()*g.numpy()))

    return stars