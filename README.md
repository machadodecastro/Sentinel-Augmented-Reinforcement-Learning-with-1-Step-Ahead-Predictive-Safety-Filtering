# Sentinel-Augmented-Reinforcement-Learning-with-1-Step-Ahead-Predictive-Safety-Filtering
Sentinel-Augmented Reinforcement Learning with 1-Step Ahead Predictive Safety Filtering

1	Introduction

Reinforcement Learning (RL) is a branch of machine learning concerned with how agents ought to take actions in an environment to maximize some notion of cumulative reward. It is inspired by behavioral psychology, where learning occurs through trial and error, with the agent receiving feedback in the form of rewards or penalties. Regarding the relationship between agent and environment, RL revolves around an agent interacting with an environment. The agent perceives the state of the environment and takes actions based on a policy to maximize rewards. The agent perceives the state of the environment and takes actions based on a policy to maximize rewards. The environment is something that the agent can observe and act on [Sutton et al., 2018].

In this work, a model-based approach to integrity and security was adopted, describing a two-agent reinforcement learning setup, where the sentinel acts as a meta-agent or advisor, sending reward signals or risk predictions to guide the main agent.

The central motivation behind this approach is that model-based methods have the ability to predict unsafe situations before they occur. In many real-world scenarios, the system designer already knows which states represent risks—for example: a robot colliding with parts of itself or surrounding objects, a vehicle entering the wrong lane, or a patient's glucose levels rising abruptly. Modelless learning methods, on the other hand, generally fail to incorporate this prior knowledge and end up needing to experience some failures or safety violations to finally learn how to avoid them.

I assume that, in practice, it is enough to predict a few steps ahead to prevent dangerous events from occurring. Consider, for example, the case illustrated in Figure 1 in which the sentinel explores the environment first. It evaluates possible actions and resulting positions relative to the target. So it sends signals (positive / negative / neutral) to the main agent, representing estimated reward or safety for each action. The main agent then uses these signals to make safer choices — avoiding costly or dangerous trial-and-error.

<img width="643" height="355" alt="image" src="https://github.com/user-attachments/assets/c8630d97-55f3-4522-b8c3-d081075e0efa" />


2	Background

In this work, a deterministic Markov Decision Process (MDP) M = (S, A, P, R, γ) is considered, where S is the state space, A the action space, P the transition dynamics, R the reward function and  γ the discount factor. 
The sentinel is an external auxiliary function and its behavior does not alter the Markov decision process (MDP).

<img width="614" height="201" alt="image" src="https://github.com/user-attachments/assets/37f74319-966b-4b99-9a7b-2839f1368c28" />

<img width="418" height="262" alt="image" src="https://github.com/user-attachments/assets/d1cbb2c9-c621-4703-a705-d0c73dc6557c" />

The sentinel does not move in the environment — instead, it simulates 1-step ahead what would happen if the main agent moved in each direction. It computes a sentinel Q-value or risk estimate for each possible action. After that, the sentinel agent broadcasts signals to the main agent:  +1, 0, -1 where +1 = positive reward, 0 = neutral reward  and -1 = negative reward.
The main agent queries the sentinel for each possible move. It chooses the move with the highest sentinel signal (i.e., least risky or most rewarding).
Key behavior:
The sentinel still inspects each possible action and returns signals +1 / 0 / -1.


The main agent maintains a Q-table and acts ε-greedily (so it learns through exploration) but the exploration cost is reduced since sentinel already discovered useful policies.


The sentinel signal is used as reward shaping: 
shaped_reward = env_reward + sentinel_weight * signal.


The agent updates its Q-table using the standard tabular Q-learning rule or using DQN (Deep Q-Network).

3	Method

In contrast to approaches that require the engineer to explicitly identify which states are recoverable or irrecoverable — a task that typically demands detailed knowledge of the system dynamics — relying on a sentinel mechanism capable of predicting one step ahead. Instead of predefined recoverability labels, the sentinel evaluates the immediate outcome of each action and flags states that are likely to lead to safety violations. Importantly that any true safety breach would occur shortly after the agent enters such a high-risk region detected by the sentinel.
So, in traditional Q-learning:

<img width="653" height="43" alt="image" src="https://github.com/user-attachments/assets/3c2b47a6-6870-4b21-b0ef-92e0dc4f26fb" />

With the sentinel signal (predictive adjustment) in Sentinel-Augmented Reinforcement Learning with One-Step Ahead:

<img width="491" height="38" alt="image" src="https://github.com/user-attachments/assets/6f4ddcee-a599-48a2-a521-883e44c5770e" />

where:
β is the weight of the sentinel's influence (e.g., 0.2 to 0.5)
signal ∈ {−1, 0, +1}

With this, for example, DQN is conditioned by the sentinel. Its agent learns the common Q function:

<img width="411" height="98" alt="image" src="https://github.com/user-attachments/assets/e0c0abfc-6012-42be-a05b-d417ed4e4016" />

But the derived policy is: 

<img width="411" height="98" alt="image" src="https://github.com/user-attachments/assets/ecef4991-7698-4acf-9388-e5bfab2ffea8" />

The sentinel forces optimal policy to obey a constrained action space. The State (S) is extended through the actions of the sentinel agent. The input vector is not just the state, but:

<img width="417" height="48" alt="image" src="https://github.com/user-attachments/assets/e8ed5f84-4792-4b87-b8b3-76a1652a3004" />

Therefore, the network learns:

<img width="417" height="48" alt="image" src="https://github.com/user-attachments/assets/d99e8a2e-4e9c-42fe-98ed-566982301fde" />

where ŝ already contains information from the immediate future. This formalizes the idea that the sentinel looks 1 step ahead and informs the agent. The standard DQN update bellow:

<img width="417" height="48" alt="image" src="https://github.com/user-attachments/assets/5c5d32cb-cf2d-433f-8c91-a11aa9b71494" />

It remains identical because:
* the sentinel does not alter the MDP (Markov Decision Process)
* it only imposes restrictions on the policy's scope of action

4	Related Work

Safe Reinforcement Learning (Safe RL) is a branch of reinforcement learning (RL) dedicated to ensuring that autonomous agents learn effective control strategies without violating safety constraints during either training or deployment. While classical RL algorithms prioritize maximizing expected cumulative reward (Sutton & Barto, 2018), they generally assume that exploration is unconstrained. In many real-world settings—robotics, autonomous driving, healthcare, finance—unsafe exploration can lead to system failures, accidents, or harmful behavior. Safe RL introduces mechanisms to mitigate or eliminate such risks.
A foundational distinction in the field is between risk-sensitive, constrained, and model-based safety mechanisms. Risk-sensitive approaches modify the learning objective to penalize high-variance or catastrophic outcomes. Early work by Howard and Matheson (1972) and more recent formalizations by Tamar et al. (2012) use risk measures such as Value at Risk (VaR) and Conditional Value at Risk (CVaR) to incorporate aversion to extreme losses. Although these techniques reduce the likelihood of dangerous decisions, they do not strictly enforce hard safety constraints.
Constrained reinforcement learning extends classical optimization by explicitly defining states or actions that must not be violated. This line of research is connected to Constraint Markov Decision Processes (CMDPs), originally formalized by Altman (1999). Algorithms like Constrained Policy Optimization (Achiam et al., 2017) ensure that policies satisfy limits on expected costs (e.g., collisions, overheating, or constraint breaches). These methods use Lagrangian duality, projections, or optimization over Lyapunov-based constraints to ensure safety throughout learning. However, constrained approaches typically remain expectation-based, meaning that they may still allow violations in rare cases.
Model-based approaches aim to predict and avoid unsafe future states by learning or using a model of system dynamics. This perspective has gained traction because predictive models allow the agent to evaluate the safety of future trajectories before committing to actions. Approaches like Model Predictive Control with RL (Aswani et al., 2013) and the Safe Model-Based RL framework by Berkenkamp et al. (2017) leverage system dynamics—often estimated using Gaussian processes—to identify safe regions of the state space and limit exploration accordingly. The advantage of such methods is that they explicitly anticipate potential failures, which is especially valuable in physical or safety-critical environments.
Another promising direction involves the notion of protective agents or shields, inspired by formal methods in control and verification. Safety shields (Alshiekh et al., 2018) intervene at runtime by filtering unsafe actions and replacing them with safe alternatives. This concept aligns with architectures where a “sentinel” or safety supervisor predicts the consequences of candidate actions and prevents transitions into unsafe states, effectively enabling safe exploration even when the primary RL agent is unaware of hazards. Architectures that combine shielding with model-based prediction provide strong guarantees because they merge formal safety with adaptive learning.
Safe RL is also closely related to research in robust RL, which focuses on handling uncertainty in dynamics or rewards. Algorithms such as Robust Adversarial RL (Tessler et al., 2019) train agents against worst-case scenarios to ensure stable performance under perturbations. Although robustness does not inherently guarantee safety, it contributes by preventing failures due to unexpected disturbances.
Despite significant advances, Safe RL faces several open challenges. High-fidelity predictive models can be expensive to obtain, especially in environments with complex or stochastic dynamics. Ensuring safety while maintaining exploration efficiency remains difficult—overly conservative constraints may prevent the agent from discovering optimal policies. Another challenge relates to scaling Safe RL methods to high-dimensional problems, such as vision-based robotic manipulation or autonomous driving.
Overall, Safe Reinforcement Learning sits at the intersection of control theory, machine learning, and formal safety guarantees. The combination of predictive modeling, constrained optimization, shielding mechanisms, and risk-aware objectives continues to evolve, bringing RL closer to safe real-world deployment.















