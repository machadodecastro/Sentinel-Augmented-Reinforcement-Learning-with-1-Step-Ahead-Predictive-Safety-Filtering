# Sentinel-Augmented-Reinforcement-Learning-with-1-Step-Ahead-Predictive-Safety-Filtering
Sentinel-Augmented Reinforcement Learning with 1-Step Ahead Predictive Safety Filtering

1	Introduction

Reinforcement Learning (RL) is a branch of machine learning concerned with how agents ought to take actions in an environment to maximize some notion of cumulative reward. It is inspired by behavioral psychology, where learning occurs through trial and error, with the agent receiving feedback in the form of rewards or penalties. Regarding the relationship between agent and environment, RL revolves around an agent interacting with an environment. The agent perceives the state of the environment and takes actions based on a policy to maximize rewards. The agent perceives the state of the environment and takes actions based on a policy to maximize rewards. The environment is something that the agent can observe and act on [Sutton et al., 2018].

In this work, a model-based approach to integrity and security was adopted, describing a two-agent reinforcement learning setup, where the sentinel acts as a meta-agent or advisor, sending reward signals or risk predictions to guide the main agent.

The central motivation behind this approach is that model-based methods have the ability to predict unsafe situations before they occur. In many real-world scenarios, the system designer already knows which states represent risks—for example: a robot colliding with parts of itself or surrounding objects, a vehicle entering the wrong lane, or a patient's glucose levels rising abruptly. Modelless learning methods, on the other hand, generally fail to incorporate this prior knowledge and end up needing to experience some failures or safety violations to finally learn how to avoid them.

I assume that, in practice, it is enough to predict a few steps ahead to prevent dangerous events from occurring. Consider, for example, the case illustrated in Figure 1 in which the sentinel explores the environment first. It evaluates possible actions and resulting positions relative to the target. So it sends signals (positive / negative / neutral) to the main agent, representing estimated reward or safety for each action. The main agent then uses these signals to make safer choices — avoiding costly or dangerous trial-and-error.


Figure 1: An illustrative example. The sentinel agent anticipates the presence of red blocks and sends a negative signal to the principal agent so that it avoids this step. The sentinel seeks to maximize reward without jeopardizing the integrity of the principal agent.
Sentinel-Augmented Reinforcement Learning with One-Step Ahead Predictive Safety Filtering simulates reinforcement learning using auxiliary feedback, similar to teacher-student Q-learning. The reward does not override the agent's integrity and, potentially, the integrity of any other external agent in the environment. In cases of extreme risk where the main agent is surrounded by potentially harmful actions, the system interrupts training. Code is made available at 
https://github.com/machadodecastro/Sentinel-Augmented-Reinforcement-Learning-with-1-Step-Ahead-Predictive-Safety-Filtering.git

2	Background
In this work, a deterministic Markov Decision Process (MDP) M = (S, A, P, R, γ) is considered, where S is the state space, A the action space, P the transition dynamics, R the reward function and  γ the discount factor. 
The sentinel is an external auxiliary function and its behavior does not alter the Markov decision process (MDP).

where:
   if the next state has a positive reward
                                      if the next state has a negative reward
                                      if the next state has neutral reward
Formally:



and

with 𝑓 being the deterministic transition function of the grid.
The sentinel does not move in the environment — instead, it simulates 1-step ahead what would happen if the main agent moved in each direction. It computes a sentinel Q-value or risk estimate for each possible action. After that, the sentinel agent broadcasts signals to the main agent:  +1, 0, -1 where +1 = positive reward, 0 = neutral reward  and -1 = negative reward.
The main agent queries the sentinel for each possible move. It chooses the move with the highest sentinel signal (i.e., least risky or most rewarding).
Key behavior:
The sentinel still inspects each possible action and returns signals +1 / 0 / -1.


The main agent maintains a Q-table and acts ε-greedily (so it learns through exploration) but the exploration cost is reduced since sentinel already discovered useful policies.


The sentinel signal is used as reward shaping: 
shaped_reward = env_reward + sentinel_weight * signal.


The agent updates its Q-table using the standard tabular Q-learning rule or using DQN (Deep Q-Network).





3	Method
In the present work, the agent does not act on the entire set A. It acts on the safe subset:

In other words, all actions whose next state would have a negative reward are filtered out. If at least one safe action exists:

The agent can only explore within:

If there is no safe course of action, then:

These notions resemble the formulations used in earlier research on safe reinforcement learning [Hans et al., 2008]. However, in contrast to approaches that require the engineer to explicitly identify which states are recoverable or irrecoverable — a task that typically demands detailed knowledge of the system dynamics — relying on a sentinel mechanism capable of predicting one step ahead. Instead of predefined recoverability labels, the sentinel evaluates the immediate outcome of each action and flags states that are likely to lead to safety violations. Importantly that any true safety breach would occur shortly after the agent enters such a high-risk region detected by the sentinel.
So, in traditional Q-learning:

With the sentinel signal (predictive adjustment) in Sentinel-Augmented Reinforcement Learning with One-Step Ahead:

where:
β is the weight of the sentinel's influence (e.g., 0.2 to 0.5)
signal ∈ {−1, 0, +1}
With this, for example, DQN is conditioned by the sentinel. Its agent learns the common Q function:


But the derived policy is: 



The sentinel forces optimal policy to obey a constrained action space. The State (S) is extended through the actions of the sentinel agent. The input vector is not just the state, but:

Therefore, the network learns:

where     already contains information from the immediate future. This formalizes the idea that the sentinel looks 1 step ahead and informs the agent. The standard DQN update bellow:

It remains identical because:
the sentinel does not alter the MDP (Markov Decision Process)
it only imposes restrictions on the policy's scope of action
But since the network input includes the sentinel signals:

It learns to associate the sentinel's signals with future outcomes.
The mathematical role of the sentinel consists of being:
An evaluator 1-step ahead: 
A stock filter: 
Part of the DQN observation area:
A policy restriction:  



4	Related Work
Safe Reinforcement Learning (Safe RL) is a branch of reinforcement learning (RL) dedicated to ensuring that autonomous agents learn effective control strategies without violating safety constraints during either training or deployment. While classical RL algorithms prioritize maximizing expected cumulative reward (Sutton & Barto, 2018), they generally assume that exploration is unconstrained. In many real-world settings—robotics, autonomous driving, healthcare, finance—unsafe exploration can lead to system failures, accidents, or harmful behavior. Safe RL introduces mechanisms to mitigate or eliminate such risks.
A foundational distinction in the field is between risk-sensitive, constrained, and model-based safety mechanisms. Risk-sensitive approaches modify the learning objective to penalize high-variance or catastrophic outcomes. Early work by Howard and Matheson (1972) and more recent formalizations by Tamar et al. (2012) use risk measures such as Value at Risk (VaR) and Conditional Value at Risk (CVaR) to incorporate aversion to extreme losses. Although these techniques reduce the likelihood of dangerous decisions, they do not strictly enforce hard safety constraints.
Constrained reinforcement learning extends classical optimization by explicitly defining states or actions that must not be violated. This line of research is connected to Constraint Markov Decision Processes (CMDPs), originally formalized by Altman (1999). Algorithms like Constrained Policy Optimization (Achiam et al., 2017) ensure that policies satisfy limits on expected costs (e.g., collisions, overheating, or constraint breaches). These methods use Lagrangian duality, projections, or optimization over Lyapunov-based constraints to ensure safety throughout learning. However, constrained approaches typically remain expectation-based, meaning that they may still allow violations in rare cases.
Model-based approaches aim to predict and avoid unsafe future states by learning or using a model of system dynamics. This perspective has gained traction because predictive models allow the agent to evaluate the safety of future trajectories before committing to actions. Approaches like Model Predictive Control with RL (Aswani et al., 2013) and the Safe Model-Based RL framework by Berkenkamp et al. (2017) leverage system dynamics—often estimated using Gaussian processes—to identify safe regions of the state space and limit exploration accordingly. The advantage of such methods is that they explicitly anticipate potential failures, which is especially valuable in physical or safety-critical environments.
Another promising direction involves the notion of protective agents or shields, inspired by formal methods in control and verification. Safety shields (Alshiekh et al., 2018) intervene at runtime by filtering unsafe actions and replacing them with safe alternatives. This concept aligns with architectures where a “sentinel” or safety supervisor predicts the consequences of candidate actions and prevents transitions into unsafe states, effectively enabling safe exploration even when the primary RL agent is unaware of hazards. Architectures that combine shielding with model-based prediction provide strong guarantees because they merge formal safety with adaptive learning.
Safe RL is also closely related to research in robust RL, which focuses on handling uncertainty in dynamics or rewards. Algorithms such as Robust Adversarial RL (Tessler et al., 2019) train agents against worst-case scenarios to ensure stable performance under perturbations. Although robustness does not inherently guarantee safety, it contributes by preventing failures due to unexpected disturbances.
Despite significant advances, Safe RL faces several open challenges. High-fidelity predictive models can be expensive to obtain, especially in environments with complex or stochastic dynamics. Ensuring safety while maintaining exploration efficiency remains difficult—overly conservative constraints may prevent the agent from discovering optimal policies. Another challenge relates to scaling Safe RL methods to high-dimensional problems, such as vision-based robotic manipulation or autonomous driving.
Overall, Safe Reinforcement Learning sits at the intersection of control theory, machine learning, and formal safety guarantees. The combination of predictive modeling, constrained optimization, shielding mechanisms, and risk-aware objectives continues to evolve, bringing RL closer to safe real-world deployment.
Model-based Reinforcement Learning Model-Based Reinforcement Learning (MBRL) is a subfield of reinforcement learning (RL) that seeks to improve sample efficiency, planning capability, and safety by explicitly using a model of the environment’s dynamics. In contrast to model-free methods—which learn value functions or policies directly from interactions—MBRL incorporates a transition model and, often, a reward model to predict future outcomes. This distinction allows MBRL methods to leverage simulated rollouts, perform lookahead planning, and prevent unsafe or suboptimal actions before they are executed (Sutton & Barto, 2018).
Central to MBRL is the idea of learning an approximate transition function

which estimates how actions modify the environment. Classical approaches such as Dyna-Q (Sutton, 1990) interleave real-world data collection with "imagined" experience generated by the model. This framework demonstrated early on that incorporating even simple models can dramatically accelerate learning.
Over time, a spectrum of MBRL approaches emerged. On one end are planning-based methods, exemplified by the Monte Carlo Tree Search (MCTS) used in AlphaGo (Silver et al., 2017), where the model is used purely for strategic planning without gradient-based policy learning. On the other end, gradient-based MBRL methods explicitly differentiate through learned models. Notable examples include PILCO, which uses Gaussian Processes for high-precision modeling (Deisenroth & Rasmussen, 2011), achieving strong sample efficiency in robotics tasks. PILCO demonstrated that accurate models enable effective planning even in continuous, noisy control systems.
In more recent years, deep learning has facilitated MBRL in high-dimensional environments such as vision-based control. World Models (Ha & Schmidhuber, 2018) introduced the idea of learning a latent-space dynamics model paired with a controller trained inside the learned world. Similarly, PlaNet (Hafner et al., 2019) and its successor Dreamer (Hafner et al., 2020) use latent dynamics models and perform policy optimization entirely in imagination. These methods illustrate the growing trend of using MBRL to achieve both sample efficiency and generalization in complex domains.
A critical advantage of MBRL is its ability to support uncertainty estimation, which is essential for robust decision-making. Techniques like probabilistic ensembles (Chua et al., 2018) enable uncertainty-aware planning and policy learning by modeling epistemic uncertainty in the environment. This allows agents to avoid overconfident predictions in poorly understood regions of the state space.
MBRL is also a key component in safe reinforcement learning, where anticipating dangerous situations is crucial. By predicting future trajectories, agents can avoid constraint violations and mitigate catastrophic failures during training and deployment. Model-based control methods, such as Model Predictive Control (MPC), are often integrated into RL frameworks to leverage both learned models and safety constraints (Williams et al., 2017).
Despite its strengths, MBRL faces several key challenges. Learning accurate models in high-dimensional and partially observable environments remains difficult. Compounding model errors during long planning horizons can degrade performance, a phenomenon known as model bias. Researchers continue to investigate approaches such as short-horizon planning, uncertainty-based penalty terms, latent-space dynamics, and hybrid model-based/model-free architectures to alleviate this problem.
Overall, Model-Based Reinforcement Learning represents a powerful paradigm that blends predictive modeling, optimal control, and deep learning. With ongoing advancements in model learning, uncertainty estimation, and planning algorithms, MBRL continues to play a central role in building RL systems that are more sample-efficient, interpretable, and suitable for real-world deployment.





5	Conclusion
The article proposes a safety-oriented extension to reinforcement learning (RL) that augments traditional RL with a sentinel module — a lightweight predictive safety filter that looks one step ahead to anticipate the outcomes of candidate actions. The core idea is that, in many real-world systems, errors or unsafe actions can be predicted by modeling only a short horizon of the environment dynamics. Rather than relying purely on model-free trial-and-error, which may experience unsafe events during exploration, the sentinel examines each possible action, simulates the immediate next state, and classifies the result as positive, neutral, or negative. This one-step lookahead enables the system to avoid unsafe or harmful states before they are encountered, reducing costly or dangerous trial-and-error, filtering out unsafe actions from the agent’s choice set, ensuring that the agent selects only from actions that are unlikely to lead to immediate negative outcomes and incorporating domain knowledge about safety violations directly into the learning process without altering the learned value targets or the underlying Markov Decision Process (MDP).
The Q-learning or Deep Q-Network (DQN) is augmented in two ways:  the state representation is enriched with the sentinel’s output and action selection is masked to avoid sentinel-flagged dangers.
The sentinel does not change the environment’s transition dynamics or reward structure; rather, it acts as a predictive safety layer that guides policy selection during training and deployment. The approach is especially suitable when the system model can predict short-term consequences, as in robotic control, autonomous driving, or other safety-critical domains. The sentinel can be viewed as a form of action masking or shielding, which prevents the agent from executing actions that would immediately lead to unacceptable outcomes. The result is a more sample-efficient, safer learning process that leverages prior knowledge of safety constraints while preserving the theoretical foundations of RL.
References
A. Hans, D. Schneegaß, A. M. Schäfer, and S. Udluft, “Safe exploration for reinforcement learning,” Proc. European Symposium on Artificial Neural Networks (ESANN), pp. 143–148, 2008.
Achiam, J., Held, D., Tamar, A., & Abbeel, P. (2017). “Constrained Policy Optimization.” Proceedings of ICML.
Alshiekh, M., Bloem, R., Ehlers, R., Könighofer, B., Niekum, S., & Topcu, U. (2018). “Safe Reinforcement Learning via Shielding.” AAAI Conference on Artificial Intelligence.
Altman, E. (1999). Constrained Markov Decision Processes. Chapman and Hall/CRC.
Aswani, A., Gonzalez, H., Sastry, S., & Tomlin, C. (2013). “Provably Safe and Robust Learning-Based Model Predictive Control.” Automatica.
Berkenkamp, F., Turchetta, M., Schoellig, A. P., & Krause, A. (2017). “Safe Model-Based Reinforcement Learning with Stability Guarantees.” Advances in Neural Information Processing Systems (NeurIPS).
Howard, R. A., & Matheson, J. E. (1972). “Risk-Sensitive Markov Decision Processes.” Management Science.
Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
Tamar, A., Di Castro, D., & Mannor, S. (2012). “Policy Gradients with Variance Related Risk Criteria.” Proceedings of ICML.
Tessler, C., Efroni, Y., & Mannor, S. (2019). “Action Robust Reinforcement Learning and Applications in Continuous Control.” Proceedings of ICML.
R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.
Chua, K., Calandra, R., McAllister, R., & Levine, S. (2018). “Deep Reinforcement Learning in a Handful of Trials Using Probabilistic Dynamics Models.” Advances in Neural Information Processing Systems (NeurIPS).
Deisenroth, M. P., & Rasmussen, C. E. (2011). “PILCO: A Model-Based and Data-Efficient Approach to Policy Search.” Proceedings of ICML.
Ha, D., & Schmidhuber, J. (2018). “World Models.” arXiv preprint arXiv:1803.10122.
Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., & Davidson, J. (2019). “Learning Latent Dynamics for Planning from Pixels (PlaNet).” Proceedings of ICML.
Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2020). “Dream to Control: Learning Behaviors by Latent Imagination.” ICLR.
Sutton, R. S. (1990). “Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming.” Proceedings of ICML.
Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.
Silver, D., et al. (2017). “Mastering the Game of Go Without Human Knowledge.” Nature.
Williams, G., Goldfain, B., Drews, P., et al. (2017). “Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving.” IEEE Robotics and Automation Letters.
