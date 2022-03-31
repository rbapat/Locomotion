from env import ConcurrentTrainingEnv

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, env, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.env_ = env
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        # print("BEFORE FIRST ROLLOUT TRAINING START)")
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # print("BEFORE ROLLOUT (ROLLOUT START)")
        pass

    def _on_step(self) -> bool:
        self.env_.render()
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        #print("POLICY UPDATE (ROLLOUT END)")

        infos = self.locals['infos'][0]
        for idx, name in enumerate(infos['names']):
            self.logger.record(f"rewards/{name}", infos['terms'][idx].item())

        self.logger.record("train/se_loss", infos['loss'])

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

# number of parallel environments to run
PARALLEL_ENVS = 32

# number of experiences to collect per parallel environment
N_STEPS = 256

# mini-batch update size of sgd
BATCH_SIZE = 1024

# number of mini-batch updates
N_EPOCHS = 5

# total number of timesteps where each collection is one timestep
TOTAL_TIMESTEPS = 20000000000

ENTROPY_COEF = 0.01

LEARNING_RATE = 3e-4

# size of buffer that is sampled during batch updates
BUFFER_SIZE = N_STEPS * PARALLEL_ENVS


def main():
    env = ConcurrentTrainingEnv(PARALLEL_ENVS, "assets", "mini_cheetah.urdf")
    cb = CustomCallback(env)

    model = PPO('MlpPolicy', env, tensorboard_log = './concurrent_training_tb/', verbose = 2, policy_kwargs = {'net_arch': [512, 256, 64]}, batch_size = BATCH_SIZE, n_steps = N_STEPS, n_epochs = N_EPOCHS, ent_coef = ENTROPY_COEF, learning_rate = LEARNING_RATE)
    model.learn(total_timesteps = TOTAL_TIMESTEPS, callback = cb)

    print("Waiting...")
    input()
    obs = env.reset()
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        print(dones)
        input()

if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()