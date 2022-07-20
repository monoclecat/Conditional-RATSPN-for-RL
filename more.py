import numpy as np
import nlopt


class ITPS:
    grad_bound = 1e-4
    value_bound = 1e-10

    def __init__(self, eta_offset, omega_offset, constrain_entropy):
        self._constrain_entropy = constrain_entropy
        self._eta_offset = eta_offset
        self._omega_offset = omega_offset

        self._eta = None
        self._omega = None
        self._grad = np.zeros(2)
        self._succ = False
        self.step = 0

    def opt_dual(self):
        self.step = 0
        opt = nlopt.opt(nlopt.LD_LBFGS, 2)
        opt.set_lower_bounds(0.0)
        opt.set_upper_bounds(1e12)
        opt.set_min_objective(self._dual)
        try:
            opt_eta_omega = opt.optimize([10.0, 10.0])
            opt_eta = opt_eta_omega[0]
            opt_omega = opt_eta_omega[1] if self._constrain_entropy else 0.0
            return opt_eta, opt_omega
        except Exception as e:
            # NLOPT somtimes throws error very close to convergence, we check for this and return preliminary result
            # if its close enough:
            # 1.Case: Gradient near 0
            # 2.Case: eta near bound and d_omega near 0
            # 3.Case: omega near bound and d_eta near 0
            if (np.sqrt(self._grad[0]**2 + self._grad[1]**2) < ITPS.grad_bound) or \
               (self._eta < ITPS.value_bound and np.abs(self._grad[1]) < ITPS.grad_bound) or \
               (self._omega < ITPS.value_bound and np.abs(self._grad[0]) < ITPS.grad_bound):
                return self._eta, self._omega
            else:
                raise e

    def _dual(self, eta_omega, grad):
        raise NotImplementedError

    @property
    def last_eta(self):
        return self._eta

    @property
    def last_omega(self):
        return self._omega

    @property
    def last_grad(self):
        return self._grad

    @property
    def success(self):
        return self._succ

    @property
    def eta_offset(self):
        return self._eta_offset

    @eta_offset.setter
    def eta_offset(self, new_eta_offset):
        self._eta_offset = new_eta_offset

    @property
    def omega_offset(self):
        return self._omega_offset

    @omega_offset.setter
    def omega_offset(self, new_omega_offset):
        self._omega_offset = new_omega_offset


class MoreGaussian(ITPS):
    def __init__(self, dim, eta_offset, omega_offset, constrain_entropy):
        super().__init__(eta_offset, omega_offset, constrain_entropy)

        self._dim = dim
        self._dual_const_part = dim * np.log(2 * np.pi)
        self._entropy_const_part = 0.5 * (self._dual_const_part + dim)
        self._min_precision = 1/1000.0  # 1/var

    @property
    def min_precision(self):
        return self._min_precision

    def more_step(self, eps, beta, old_dist, reward_surrogate):
        self._eps = eps
        self._beta = beta
        self._succ = False

        self._old_lin_term = old_dist.lin_term
        self._old_precision = old_dist.precision
        self._old_mean = old_dist.mean
        self._old_chol_precision_t = old_dist.chol_precision.T

        self._reward_lin_term = reward_surrogate.lin_term
        self._reward_quad_term = reward_surrogate.quad_term

        old_logdet = old_dist.covar_logdet()
        self._old_term = -0.5 * (np.dot(self._old_lin_term, self._old_mean) + self._dual_const_part + old_logdet)
        self._kl_const_part = old_logdet - self._dim

        try:
            opt_eta, opt_omega = self.opt_dual()
            new_lin, new_precision = self._new_params(opt_eta + self._eta_offset, opt_omega + self._omega_offset)
            new_precision = np.maximum(new_precision, np.ones_like(new_precision) * self._min_precision)
            new_covar = np.linalg.inv(new_precision)
            new_mean = new_covar @ new_lin
            self._succ = True
            return new_mean, new_covar
        except Exception:
            self._succ = False
            return None, None

    def _new_params(self, eta, omega):
        new_lin = (eta * self._old_lin_term + self._reward_lin_term) / (eta + omega)
        new_precision = (eta * self._old_precision + self._reward_quad_term) / (eta + omega)
        return new_lin, new_precision

    def _dual(self, eta_omega, grad):
        self.step += 1
        eta = eta_omega[0] if eta_omega[0] > 0.0 else 0.0
        omega = eta_omega[1] if self._constrain_entropy and eta_omega[1] > 0.0 else 0.0
        self._eta = eta
        self._omega = omega

        eta_off = eta + self._eta_offset
        omega_off = omega + self._omega_offset

        new_lin, new_precision = self._new_params(eta_off, omega_off)
        try:
            new_covar = np.linalg.inv(new_precision)
            new_chol_covar = np.linalg.cholesky(new_covar)

            new_mean = new_covar @ new_lin
            new_logdet = 2 * np.sum(np.log(np.diagonal(new_chol_covar) + 1e-25))

            dual = eta * self._eps - omega * self._beta + eta_off * self._old_term
            dual += 0.5 * (eta_off + omega_off) * (self._dual_const_part + new_logdet + np.dot(new_lin, new_mean))

            trace_term = np.sum(np.square(self._old_chol_precision_t @ new_chol_covar))
            kl = self._kl_const_part - new_logdet + trace_term
            diff = self._old_mean - new_mean
            kl = 0.5 * (kl + np.sum(np.square(self._old_chol_precision_t @ diff)))

            grad[0] = self._eps - kl
            grad[1] = (self._entropy_const_part + 0.5 * new_logdet - self._beta) if self._constrain_entropy else 0.0
            self._grad[:] = grad
            return dual.item()

        except np.linalg.LinAlgError as e:
            grad[0] = -1.0
            grad[1] = 0.0
            return 1e12


class RepsCategorical(ITPS):

    def reps_step(self, eps, beta, old_dist, rewards):
        self._eps = eps
        self._beta = beta
        self._old_log_prob = old_dist.log_probabilities
        self._rewards = rewards

        try:
            opt_eta, opt_omega = self.opt_dual()
            new_params = self._new_params(opt_eta + self._eta_offset, opt_omega + self._omega_offset)
            self._succ = True
            return new_params
        except Exception:
            self._succ = False
            return None

    def _new_params(self, eta, omega):
        omega = omega if self._constrain_entropy else 0.0
        new_params = np.exp((eta * self._old_log_prob + self._rewards) / (eta + omega))
        if np.isnan(new_params).any():
            print(21)
        return new_params / np.sum(new_params)

    def _dual(self, eta_omega, grad):
        self.step += 1
        eta = eta_omega[0] if eta_omega[0] > 0.0 else 0.0
        omega = eta_omega[1] if self._constrain_entropy and eta_omega[1] > 0.0 else 0.0
        self._eta = eta
        self._omega = omega

        eta_off = eta + self._eta_offset
        omega_off = omega + self._omega_offset

        t1 = (eta_off * self._old_log_prob + self._rewards) / (eta_off + omega_off)
        #  one times(eta + omega) in denominator  missing
        t1_de = (omega_off * self._old_log_prob - self._rewards) / (eta_off + omega_off)
        #  t1_do = -t1 with one times (eta+omega) in denominator missing

        t1_max = np.max(t1)
        exp_t1 = np.exp(t1 - t1_max)
        sum_exp_t1 = np.sum(exp_t1) + 1e-25
        t2 = t1_max + np.log(sum_exp_t1)

        #  factor of exp(t1_max) is still missing in sum_exp_t1
        inv_sum = (1 / sum_exp_t1)
        #  missing factors of exp(t1_max) in both inv_sum and exp_t1, cancel out here.
        t2_de =   inv_sum * np.sum(t1_de * exp_t1)
        t2_do = - inv_sum * np.sum(t1    * exp_t1)  #  -t2 =  t2_do

        grad[0] = self._eps + t2 + t2_de  # missing factor in t2_de cancels out with missing factor here
        #  missing factor in t2_do cancels out with missing factor here
        grad[1] = - self._beta + t2 + t2_do if self._constrain_entropy else 0.0

        self._grad[:] = grad

        dual = eta * self._eps - omega * self._beta + (eta_off + omega_off) * t2
        return dual


class Gaussian:

    def __init__(self, mean, covar):
        self._dim = mean.shape[-1]
        self.update_parameters(mean, covar)

    def density(self, samples):
        return np.exp(self.log_density(samples))

    def log_density(self, samples):
        norm_term = self._dim * np.log(2 * np.pi) + self.covar_logdet()
        diff = samples - self._mean
        exp_term = np.sum(np.square(diff @ self._chol_precision), axis=-1)
        return -0.5 * (norm_term + exp_term)

    def log_likelihood(self, samples):
        return np.mean(self.log_density(samples))

    def sample(self, num_samples):
        eps = np.random.normal(size=[num_samples, self._dim])
        return self._mean + eps @ self._chol_covar.T

    def entropy(self):
        return 0.5 * (self._dim * np.log(2 * np.pi * np.e) + self.covar_logdet())

    def kl(self, other):
        trace_term = np.sum(np.square(other.chol_precision.T @ self._chol_covar))
        kl = other.covar_logdet() - self.covar_logdet() - self._dim + trace_term
        diff = other.mean - self._mean
        kl = kl + np.sum(np.square(other.chol_precision.T @ diff))
        return 0.5 * kl

    def covar_logdet(self):
        return 2 * np.sum(np.log(np.diagonal(self._chol_covar) + 1e-25))

    def update_parameters(self, mean, covar):
        try:
            chol_covar = np.linalg.cholesky(covar)
            inv_chol_covar = np.linalg.inv(chol_covar)
            precision = inv_chol_covar.T @ inv_chol_covar

            self._chol_precision = np.linalg.cholesky(precision)
            self._mean = mean
            self._lin_term = precision @ mean
            self._covar = covar
            self._precision = precision

            self._chol_covar = chol_covar

        except Exception as e:
            print("Gaussian Paramameter update rejected:", e)

    @property
    def mean(self):
        return self._mean

    @property
    def covar(self):
        return self._covar

    @property
    def lin_term(self):
        return self._lin_term

    @property
    def precision(self):
        return self._precision

    @property
    def chol_covar(self):
        return self._chol_covar

    @property
    def chol_precision(self):
        return self._chol_precision


class Categorical:

    def __init__(self, probabilities):
        self._p = probabilities

    def sample(self, num_samples):
        thresholds = np.expand_dims(np.cumsum(self._p), 0)
        thresholds[0, -1] = 1.0
        eps = np.random.uniform(size=[num_samples, 1])
        samples = np.argmax(eps < thresholds, axis=-1)
        return samples

    @property
    def probabilities(self):
        return self._p

    @probabilities.setter
    def probabilities(self, new_probabilities):
        self._p = new_probabilities

    @property
    def log_probabilities(self):
        return np.log(self._p + 1e-25)

    def entropy(self):
        return - np.sum(self._p * np.log(self._p + 1e-25))

    def kl(self, other):
        return np.sum(self._p * (np.log(self._p + 1e-25) - other.log_probabilities))


class RegressionFunc:

    def __init__(self, reg_fact, normalize, unnormalize_output, bias_entry=None):
        self._reg_fact = reg_fact
        self._normalize = normalize
        self._unnormalize_output = unnormalize_output
        self._bias_entry = bias_entry
        self._params = None
        self.o_std = None

    def __call__(self, inputs):
        if self._params is None:
            raise AssertionError("Model not trained yet")
        return self._feature_fn(inputs) @ self._params

    def _feature_fn(self, x):
        raise NotImplementedError

    def _normalize_features(self, features):
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        # do not normalize bias
        if self._bias_entry is not None:
            mean[self._bias_entry] = 0.0
            std[self._bias_entry] = 1.0
        features = (features - mean) / std
        return features, np.squeeze(mean, axis=0), np.squeeze(std, axis=0)

    def _normalize_outputs(self, outputs):
        mean = np.mean(outputs)
        std = np.std(outputs)
        outputs = (outputs - mean) / std
        return outputs, mean, std

    def _undo_normalization(self, params, f_mean, f_std, o_mean, o_std):
        if self._unnormalize_output:
            params *= (o_std / f_std)
            params[self._bias_entry] = params[self._bias_entry] - np.dot(params, f_mean) + o_mean
        else:
            params *= (1.0 / f_std)
            params[self._bias_entry] = params[self._bias_entry] - np.dot(params, f_mean)
        return params

    def fit(self, inputs, outputs, weights=None):
        if len(outputs.shape) > 1:
            outputs = np.squeeze(outputs)
        features = self._feature_fn(inputs)
        if self._normalize:
            features, f_mean, f_std = self._normalize_features(features)
            outputs, o_mean, o_std = self._normalize_outputs(outputs)

        if weights is not None:
            if len(weights.shape) == 1:
                weights = np.expand_dims(weights, 1)
            weighted_features = weights * features
        else:
            weighted_features = features

        #regression
        reg_mat = np.eye(weighted_features.shape[-1]) * self._reg_fact
        if self._bias_entry is not None:
            reg_mat[self._bias_entry, self._bias_entry] = 0.0
        try:
            self._params = np.linalg.solve(weighted_features.T @ features + reg_mat, weighted_features.T @ outputs)
            if self._normalize:
                self._undo_normalization(self._params, f_mean, f_std, o_mean, o_std)
                self.o_std = o_std
        except np.linalg.LinAlgError as e:
            print("Error during matrix inversion", e.what())


class LinFunc(RegressionFunc):

    def __init__(self, reg_fact, normalize, unnormalize_output):
        super().__init__(reg_fact, normalize, unnormalize_output, -1)

    def _feature_fn(self, x):
        return np.concatenate([x, np.ones([x.shape[0], 1], dtype=x.dtype)], 1)


class QuadFunc(RegressionFunc):
    #*Fits - 0.5 * x ^ T  Rx + x ^ T r + r_0 ** * /

    def __init__(self, reg_fact, normalize, unnormalize_output):
        super().__init__(reg_fact, normalize, unnormalize_output, bias_entry=-1)
        self.quad_term = None
        self.lin_term = None
        self.const_term = None
        self._square_feat_upper_tri_ind = None

    def __square_feat_upper_tri_ind(self, n):
        if self._square_feat_upper_tri_ind is None or self._square_feat_upper_tri_ind.shape[0] != n:
            self._square_feat_upper_tri_ind = np.triu_indices(n)
        return self._square_feat_upper_tri_ind

    def _feature_fn(self, x):
        lin_feat = x
        quad_feat = np.transpose((x[:, :, np.newaxis] @ x[:, np.newaxis, :]), [1, 2, 0])
        quad_feat = quad_feat[self.__square_feat_upper_tri_ind(x.shape[-1])].T
        features = np.hstack([quad_feat, lin_feat, np.ones([x.shape[0], 1])])
        return features

    def fit(self, inputs, outputs, weights=None, sample_mean=None, sample_chol_cov=None):
        if sample_mean is None:
            assert sample_chol_cov is None
        if sample_chol_cov is None:
            assert sample_mean is None

        #whithening
        if sample_mean is not None and sample_chol_cov is not None:
            inv_samples_chol_cov = np.linalg.inv(sample_chol_cov)
            inputs = (inputs - sample_mean) @ inv_samples_chol_cov.T

        dim = inputs.shape[-1]

        super().fit(inputs, outputs, weights)

        idx = np.triu(np.ones([dim, dim], np.bool))

        qt = np.zeros([dim, dim])
        qt[idx] = self._params[:- (dim + 1)]
        self.quad_term = - qt - qt.T

        self.lin_term = self._params[-(dim + 1): -1]
        self.const_term = self._params[-1]

        #unwhitening:
        if sample_mean is not None and sample_chol_cov is not None:
            self.quad_term = inv_samples_chol_cov.T @ self.quad_term @ inv_samples_chol_cov
            t1 = inv_samples_chol_cov.T @ self.lin_term
            t2 = self.quad_term @ sample_mean
            self.lin_term = t1 + t2
            self.const_term += np.dot(sample_mean, -0.5 * t2 - t1)
