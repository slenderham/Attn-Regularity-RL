import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive


class UFRL:
    def __init__(self):
        self.num_params = 5
        self.param_bounds = jnp.array([[-5, 5], [0, 100], [0, 1], [0, 1], [0, 1]])


    def forward(self, data, params):
        bias = params[0] # side bias
        beta = params[1] # inverse temperature
        alpha_r = params[2] # positive learning rate
        alpha_n = params[3] # positive learning rate
        decay = params[4] # decay

        num_dims = data['num_dims']
        num_vals = data['num_vals']

        num_trials = len(data['rewards'])

        values = jnp.zeros((num_dims, num_vals))

        # for t in range(num_trials):
        #     stims = data['stimuli']

        
        return

    def fit(self, data):
        # Number of subjects
        n_subj = len(data)
        # parameter mean
        param_mu = numpyro.sample("param_mu", dist.Uniform(self.param_bounds[:,0], self.param_bounds[:,1]))
        # parameter std
        param_sigma = numpyro.sample("param_sigma", dist.HalfCauchy(100*jnp.ones(self.num_params)))
        # Plate for the subjects
        with numpyro.plate("subject", n_subj):
            params = numpyro.sample("param_sbj", \
                dist.TruncatedDistribution(
                    dist.TransformedDistribution(
                        dist.Normal(jnp.zeros_like(param_mu), jnp.ones_like(param_sigma)),
                        dist.transforms.AffineTransform(param_mu, param_sigma),
                    ), self.param_bounds[:,0], self.param_bounds[:,1]))

        with numpyro.plate("subject", n_subj) as ind:
            numpyro.sample("obs", dist.Bernoulli(logits=log_probs), obs=data[ind].choice)

        return

if __name__=='__main__':
    ufrl_model = UFRL()
    nuts_kernel = NUTS(ufrl_model.fit)
    mcmc = MCMC(nuts_kernel, num_samples=2000, num_warmup=2000)
    rng_key = jnp.random.PRNGKey(0)
    mcmc.run(rng_key, data)
    posterior_samples = mcmc.get_samples()