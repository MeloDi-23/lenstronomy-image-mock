import numpy as np
import yaml
from functools import partial
class ParameterManager:
    """
    Manages multiparameter lnprob function to make it fit into the emcee.
    support: manage prior(and add flat prior to lnprob function), default value, fix a specific value, read configs from yaml file.
    yaml file should be formatted as:
    Name:
        init: xxx
        free: true|false     # default as false
        prior: [xxx, xxx]    # default as no prior
    
    if you set kwargs to be true, parameters will be warped and passed to it as lnprob(name1=value1, name2=value2, ...)
    otherwise the parameter will be passed directly as lnprob(value1, value2, ...), 
    in this case the parameter name doesn't matter, but order of parameters matters
    """
    @staticmethod
    def from_yaml(file, *args, **kwargs):
        to_close = False
        if type(file) == str:
            file = open(file, 'r')
            to_close = True
        config = yaml.load(file, yaml.FullLoader)
        parameter_names = []
        free_parameter = []
        initial_state = []
        prior = []
        for name, v in config.items():
            if ('init' not in v) or (type(v['init']) not in [int, float]):
                raise KeyError(f'No init or invalid init for parameter {name}')
            v.setdefault('free', False)
            v.setdefault('prior', [-np.inf, np.inf])
            if len(v['prior']) != 2:
                raise ValueError(f'Invalid prior for parameter {name}')
            parameter_names.append(name)
            free_parameter.append(v['free'])
            initial_state.append(v['init'])
            prior.append(v['prior'])
        n_para = len(parameter_names)
        initial_state = np.array(initial_state, float)
        free_parameter = np.array(free_parameter, bool)
        prior = np.array(prior, float)

        if to_close:
            file.close()

        assert prior.shape == (n_para, 2)
        assert initial_state.shape == (n_para, )

        return ParameterManager(parameter_names, free_parameter, initial_state, prior, *args, **kwargs)

    def __init__(self, parameter_names: list[str], free_parameter, initial_state, prior, lnprob: callable, use_kwargs=True):
        self.parameter_names = parameter_names
        self.n_para = len(self.parameter_names)
        self.free_parameter = np.zeros(self.n_para, bool)
        self.free_parameter[:] = free_parameter
        self.initial_state = np.zeros(self.n_para, float)
        self.initial_state[:] = initial_state

        self.prior = prior
        self.lnprob = lnprob
        self.check_init()

    def check_init(self):
        for i in np.where(self.free_parameter)[0]:            
            if not (self.initial_state[i] <= self.prior[i,1] and self.initial_state[i] >= self.prior[i,0]):
                raise ValueError('{}\' s initial value {:g} is beyond prior [{:g}, {:g}]'.format(
                    self.parameter_names[i], self.initial_state[i], self.prior[i,0], self.prior[i,1]
                ))

    @property
    def ndim(self):
        return self.free_parameter.sum()
    
    def random_init_state(self, nwalker, max_scatter=np.inf):
        ndim = self.ndim
        initial_value = np.zeros((nwalker, ndim), float)
        index = 0
        init = self.initial_state[self.free_parameter]
        prior = self.prior[self.free_parameter,:]

        width = np.minimum(np.minimum(init - prior[:,0], prior[:,1] - init)/2, max_scatter)
        for i in range(ndim):
            initial_value[:,i] = init[i] + np.random.uniform(-width[i], width[i], nwalker)

        return initial_value

    def __repr__(self):
        name = 'ParameterManager'
        format = '{}: {:g}'
        fixed_fmt = '[fixed]'
        range_fmt = '[{:g}-{:g}]'
        joiner = '\n'+' '*len(name)
        content = []
        for i in range(self.n_para):
            format.format(self.parameter_names[i], self.initial_state[i])
            if self.free_parameter[i]:
                format += range_fmt.format(*self.prior[i])
            else:
                format += fixed_fmt
        return name + '(' + joiner.join(content) + ')'

    def within_prior(self, full_para):
        return ((full_para <= self.prior[:,1]) & (full_para >= self.prior[:,0])).all()

    def lnlikely(self, parameters):
        full_para = self.initial_state.copy()
        full_para[self.free_parameter] = parameters
        if self.within_prior(full_para):
            return self.lnprob(**dict(zip(self.parameter_names, full_para)))
        return -np.inf
