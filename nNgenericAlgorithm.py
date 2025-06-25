import numpy as np
import argparse
import csv

# -------------------- Aktivacijska funkcija --------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# -------------------- Neuronska mreža --------------------
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            w = np.random.normal(0, 0.01, (layer_sizes[i+1], layer_sizes[i]))
            b = np.random.normal(0, 0.01, (1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def set_weights(self, flat_weights):
        # Postavi sve težine i biasove iz 1D vektora
        pointer = 0 
        for i in range(self.num_layers):
            w_shape = self.weights[i].shape
            b_shape = self.biases[i].shape
            w_size = w_shape[0] * w_shape[1]
            b_size = b_shape[1]
            self.weights[i] = flat_weights[pointer:pointer+w_size].reshape(w_shape)
            pointer += w_size
            self.biases[i] = flat_weights[pointer:pointer+b_size].reshape(b_shape)
            pointer += b_size

    def broj_tezina_I_biaseva(self):
        # Vrati sve težine i biasove kao 1D vektor
        flat = []
        for w, b in zip(self.weights, self.biases):
            flat.append(w.flatten())
            flat.append(b.flatten())
        return np.concatenate(flat)

    def forward(self, X):
        a = X
        for i in range(self.num_layers):
            z = np.dot(a, self.weights[i].T) + self.biases[i]
            if i < self.num_layers - 1:
                a = sigmoid(z)
            else:
                a = z  # Nema aktivacije na izlazu
        return a

# -------------------- MSE --------------------
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# -------------------- Učitavanje podataka --------------------
def load_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = np.array([[float(x) for x in row] for row in reader])
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)
    return X, y

# -------------------- Genetski algoritam --------------------
def genetic_algorithm(
    nn_arch, X_train, y_train, X_test, y_test,
    popsize, elitism, p, K, iterations, print_every=2000
):
    # Inicijalizacija populacije
    nn = NeuralNetwork(nn_arch)
    num_params = nn.get_weights().shape[0]


    population = [np.random.normal(0, 0.01, num_params) for _ in range(popsize)]

    def fitness(weights):
        nn.set_weights(weights)
        y_pred = nn.forward(X_train)
        err = mean_squared_error(y_train, y_pred)
        return -err  # Veća fitness je bolja, a manji err je bolji

    best_errs = []
    for gen in range(1, iterations+1):
        # Izračunaj fitness za sve
        fits = np.array([fitness(w) for w in population])
        errs = -fits
        # Elitizam
        elite_idx = fits.argsort()[::-1][:elitism]
        new_population = [population[i].copy() for i in elite_idx]
        # Fitness proportional selection
        fits_shifted = fits - fits.min() + 1e-8  # sve pozitivno
        probs = fits_shifted / fits_shifted.sum()
        while len(new_population) < popsize:
            # Selektiraj roditelje
            parents_idx = np.random.choice(popsize, 2, p=probs, replace=False)
            p1, p2 = population[parents_idx[0]], population[parents_idx[1]]
            # Križanje (aritmetička sredina)
            child = (p1 + p2) / 2
            # Mutacija
            mask = np.random.rand(num_params) < p
            child[mask] += np.random.normal(0, K, np.sum(mask))
            new_population.append(child)
        population = new_population[:popsize]
        # Ispis svakih print_every generacija
        if gen % print_every == 0 or gen == iterations:
            best_idx = np.argmax(fits)
            nn.set_weights(population[best_idx])
            y_pred = nn.forward(X_train)
            err = mean_squared_error(y_train, y_pred)
            print(f"[Train error @{gen}]: {err:.6f}")
            best_errs.append(err)
    # Test error na najboljoj jedinki
    fits = np.array([fitness(w) for w in population])
    best_idx = np.argmax(fits)
    nn.set_weights(population[best_idx])
    y_pred = nn.forward(X_test)
    test_err = mean_squared_error(y_test, y_pred)
    print(f"[Test error]: {test_err:.6f}")

# -------------------- Main --------------------
def parse_nn_arch(s, input_dim):
    if s == '5s':
        return [input_dim, 5, 1]
    elif s == '20s':
        return [input_dim, 20, 1]
    elif s == '5s5s':
        return [input_dim, 5, 5, 1]
    else:
        raise ValueError('Nepoznata arhitektura mreže!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--nn', required=True)
    parser.add_argument('--popsize', type=int, required=True)
    parser.add_argument('--elitism', type=int, required=True)
    parser.add_argument('--p', type=float, required=True)
    parser.add_argument('--K', type=float, required=True)
    parser.add_argument('--iter', type=int, required=True)
    args = parser.parse_args()

    X_train, y_train = load_csv(args.train)
    X_test, y_test = load_csv(args.test)
    input_dim = X_train.shape[1]
    nn_arch = parse_nn_arch(args.nn, input_dim)

    genetic_algorithm(
        nn_arch, X_train, y_train, X_test, y_test,
        args.popsize, args.elitism, args.p, args.K, args.iter
    )

if __name__ == '__main__':
    main() 