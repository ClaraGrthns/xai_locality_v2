# %%
import numpy as np
from scipy.special import gamma
from scipy.integrate import quad
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

random_seed = 42
np.random.seed(random_seed)
import numpy as np
from scipy import stats
from scipy.integrate import nquad, quad
from functools import partial
from  scipy.stats import chi2, ncx2

def generate_datasets(d, n1=50, n2=5000):
    D1 = np.random.multivariate_normal(np.zeros(d), np.eye(d), n1)
    D2 = np.random.multivariate_normal(np.zeros(d), np.eye(d), n2)
    return D1, D2

def generate_random_point_in_ball(center, radius, d):
    direction = np.random.randn(d)# Generate random direction
    direction = direction / np.linalg.norm(direction)
    r = radius * np.random.random()#**(1/d) # Generate random radius
    return center + r * direction

def create_centered_linear_classifier(center, d):
    """Create a random linear classifier through the center point"""
    normal_vector = np.random.randn(d)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    bias = -np.dot(normal_vector, center)
    return normal_vector, bias 

def linear_classifier(x, normal_vector, bias):
    return np.dot(x, normal_vector) + bias > 0

def create_linear_classifier(center, point, d):
    """Create a linear use point - center as normal vector and create linear classifier"""
    if np.allclose(center, point):
        return create_centered_linear_classifier(center, d)
    normal_vector = point - center
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    bias = -np.dot(normal_vector, point)
    return normal_vector, bias 

def integral_cap_mc(n_samples, dimension, radius, center, mean, l_classifier):
    points = np.random.uniform(-radius, radius, (n_samples, dimension)) + center
    def prob_in_cap(x): 
        in_ball = np.linalg.norm(x-center, axis=1) <= radius
        in_ball_cap = in_ball * l_classifier(x)
        return stats.multivariate_normal.pdf(x, mean = mean, cov=1) * in_ball_cap
    return np.mean(prob_in_cap(points)) * (2*radius)**dimension

def empirical_accuracy(center, points, tree, radius, l_classifier):
    indices = tree.query_radius([center], r=radius)[0]
    ball_points = points[indices]
    if len(ball_points) == 0:
        return 0, 0
    predictions = l_classifier(ball_points)
    return np.mean(predictions), len(ball_points)

def _process_point(x, D2, tree, radius, mean, dimension, n_samples_mc):
    normal_vector, bias = create_centered_linear_classifier(x, dimension)
    l_classifier = partial(linear_classifier, normal_vector=normal_vector, bias=bias)
    mass_cap = integral_cap_mc(n_samples_mc, dimension, radius, x, mean, l_classifier)
    mass_ball = ncx2.cdf(radius**2, dimension, np.linalg.norm(x)**2) 
    theo_acc = mass_cap/mass_ball
    if theo_acc > 1:
        print(f"theo_acc: {theo_acc}, mass_cap: {mass_cap}, mass_ball: {mass_ball}")
    emp_acc, n_points_in_ball_per_x = empirical_accuracy(x, D2, tree, radius, l_classifier)
    return theo_acc, emp_acc, n_points_in_ball_per_x

def compute_estimates(D1, D2, dimension, radius, n_samples_mc, mean, random_hyperplane=False):
    theoretical_estimates = []
    empirical_estimates = []
    tree = BallTree(D2)
    for x in D1:
        theo_acc, emp_acc, n_points_in_ball_per_x = _process_point(x, D2, tree, radius, mean, dimension, n_samples_mc)
        theoretical_estimates.append(theo_acc)
        empirical_estimates.append(emp_acc)

    avg_theoretical = np.mean(theoretical_estimates)
    avg_empirical = np.mean(empirical_estimates)
    se_empirical = (np.array(theoretical_estimates) - np.array(empirical_estimates))**2
    mse_empirical = np.mean(se_empirical)
    return avg_theoretical, avg_empirical, mse_empirical, theoretical_estimates, empirical_estimates, se_empirical

random_seed = 42
np.random.seed(random_seed)
n_samples_mc=10_000_000
n_samples_d1 = 50
radius = 5
ls_n_samples_d2 = np.linspace(500, 500_000, 30, dtype=int)
ls_theoretical_estimates, ls_empirical_estimates, ls_se = [], [], []
ls_avg_empirical, ls_avg_theoretical, ls_mse = [], [], []
radius = 5
dimension = 10
mean = np.zeros(dimension)

for n_samples_d2 in ls_n_samples_d2:
    D1, D2 = generate_datasets(dimension, n_samples_d1, n_samples_d2)
    avg_theoretical, avg_empirical, mse_empirical, theoretical_estimates, empirical_estimates, se_empirical = compute_estimates(D1, D2, dimension, radius, n_samples_mc, mean)
    ls_avg_theoretical.append(avg_theoretical)
    ls_avg_empirical.append(avg_empirical)
    ls_se.append(se_empirical)
    ls_empirical_estimates.append(empirical_estimates)
    ls_theoretical_estimates.append(theoretical_estimates)
    ls_mse.append(mse_empirical)

ls_theoretical_estimates = np.array(ls_theoretical_estimates)  # Shape (10, 50)
ls_empirical_estimates = np.array(ls_empirical_estimates)      # Shape (10, 50)
ls_se = np.array(ls_se)                                        # Shape (10, 50)
ls_avg_theoretical = np.array(ls_avg_theoretical).reshape(-1, 1)  # Shape (10, 1)
ls_avg_empirical = np.array(ls_avg_empirical).reshape(-1, 1)      # Shape (10, 1)
ls_mse = np.array(ls_mse).reshape(-1, 1) 

print("ls_theoretical_estimates.shape, ls_avg_empirical.shape, ls_se.shape, ls_avg_theoretical.shape, ls_avg_empirical.shape, ls_mse.shape ")                        # Shape (10, 1)
print(ls_theoretical_estimates.shape, ls_avg_empirical.shape, ls_se.shape, ls_avg_theoretical.shape, ls_avg_empirical.shape, ls_mse.shape )                        # Shape (10, 1)


np.save(f'ls_theoretical_estimates_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_theoretical_estimates)
np.save(f'ls_empirical_estimates_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_empirical_estimates)
np.save(f'ls_se_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_se)
np.save(f'ls_mse_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_mse)
np.save(f'ls_avg_empirical_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_avg_empirical)
np.save(f'ls_avg_theoretical_radius{radius}_dim_{dimension}_nd2{ls_n_samples_d2[0]}-{ls_n_samples_d2[-1]}.npy', ls_avg_theoretical)




random_seed = 42
np.random.seed(random_seed)
n_samples_mc=1_000_000
n_samples_d1 = 50
n_samples_d2 = 50_000
dimensions = np.linspace(2, 40, 10, dtype=int)
ls_theoretical_estimates, ls_empirical_estimates, ls_se = [], [], []
ls_avg_empirical, ls_avg_theoretical, ls_mse = [], [], []
radius = 5

for dimension in dimensions:
    mean = np.zeros(dimension)
    D1, D2 = generate_datasets(dimension, n_samples_d1, n_samples_d2)
    avg_theoretical, avg_empirical, mse_empirical, theoretical_estimates, empirical_estimates, se_empirical = compute_estimates(D1, D2, dimension, radius, n_samples_mc, mean)
    ls_theoretical_estimates.append(theoretical_estimates)
    ls_avg_theoretical.append(avg_theoretical)
    ls_avg_empirical.append(avg_empirical)
    ls_empirical_estimates.append(empirical_estimates)
    ls_se.append(mse_empirical)
    ls_mse.append(mse_empirical)

ls_theoretical_estimates = np.array(ls_theoretical_estimates)  # Shape (10, 50)
ls_empirical_estimates = np.array(ls_empirical_estimates)      # Shape (10, 50)
ls_se = np.array(ls_se)                                        # Shape (10, 50)
ls_avg_theoretical = np.array(ls_avg_theoretical).reshape(-1, 1)  # Shape (10, 1)
ls_avg_empirical = np.array(ls_avg_empirical).reshape(-1, 1)      # Shape (10, 1)
ls_mse = np.array(ls_mse).reshape(-1, 1) 

np.save(f'ls_theoretical_estimates_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_theoretical_estimates)
np.save(f'ls_empirical_estimates_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_empirical_estimates)
np.save(f'ls_se_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_se)
np.save(f'ls_mse_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_mse)
np.save(f'ls_avg_empirical_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_avg_empirical)
np.save(f'ls_avg_theoretical_radius{radius}_dim_{dimensions[0]}-{dimensions[-1]}_nd2{n_samples_d2}.npy', ls_avg_theoretical)