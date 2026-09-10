# 0. Reinforcement Learning Background

> Source: [https://rlgym.org/Cheatsheets/reinforcement_learning_terms/](https://rlgym.org/Cheatsheets/reinforcement_learning_terms/)  
> Scraped for local reference from the public RLGym docs.

# 0\. Reinforcement Learning Background  
  
What follows is a series of definitions that may be useful to understand the concepts of reinforcement learning when training an agent in an RLGym environment. Note that these definitions are not meant to be exhaustive, and we will formulate the reinforcement learning setting in a somewhat non-standard way to better align with the environments typically considered by practitioners using RLGym.

## 1\. The Basics​

A _decision process_ P\mathcal{P}P, sometimes called an _environment_ , is characterized by a set of _states_ S\mathcal{S}S, a set of _actions_ A\mathcal{A}A, and a state transition probability function T(s′∣s,a)\mathcal{T}(s' \mid s, a)T(s′∣s,a). To interact with an environment, an agent must follow a _policy_ π∈Π\pi \in \Piπ∈Π, where Π\PiΠ denotes the set of admissible policies.

When an agent executes an _action_ a∈Aa \in \mathcal{A}a∈A in state s∈Ss \in \mathcal{S}s∈S, the environment transitions to a new state s′s's′ according to T(s′∣s,a)\mathcal{T}(s' \mid s, a)T(s′∣s,a).

For our purposes it is useful to further consider a set of _observations_ O\mathcal{O}O, which are representations of states that an agent acts upon. The observation function O:S→O\mathbf{O} : \mathcal{S} \rightarrow \mathcal{O}O:S→O maps states sss to observations ooo.

A _policy_ π:R∣O∣→R∣A∣\pi : \mathbb{R}^{\mid\mathcal{O}\mid} \rightarrow \mathbb{R}^{\mid\mathcal{A}\mid}π:R∣O∣→R∣A∣ is a function that maps an observation ooo to a real-valued vector, π(o)∈R∣A∣\pi(o) \in \mathbb{R}^{\mid\mathcal{A}\mid}π(o)∈R∣A∣.

In practice, a policy typically defines a distribution over actions. For discrete actions, π\piπ directly outputs the probability mass for each action. For continuous actions, the most common approach is for the policy to parameterize some known distribution (e.g. a Gaussian distribution) rather than try to directly learn the density of the action space. This is called the [reparameterization trick](<https://en.wikipedia.org/wiki/Reparameterization_trick>).

An _action function_ I:R∣A∣→A\mathbf{I} : \mathbb{R}^{\mid\mathcal{A}\mid} \rightarrow \mathcal{A}I:R∣A∣→A is a function that maps the output of a policy to an action.

The complete state-to-action mapping is given by a=I(π(O(s)))a = \mathbf{I}(\pi(\mathbf{O}(s)))a=I(π(O(s))). We will refer to this mapping as an _agent_.

Following each action aaa in state sss, a _reward function_ R:S×A→R\mathbf{R} : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}R:S×A→R generates a scalar reward rrr.

We refer to a single interaction between the agent and the environment as a _timestep_ , which contains (s,a,r,s′)(s, a, r, s')(s,a,r,s′).

### 1.1 Trajectories and Returns​

We are concerned with two types of sequences:

  * A _trajectory_ : Any sequence of timesteps from some state sts_tst​ to another state st+ns_{t+n}st+n​.
  * An _episode_ : A special case of trajectory that begins with an initial state s0s_0s0​ and ends with a terminal state sTs_TsT​.

We will denote such sequences of timesteps as τ=(st,at,rt,st+1,at+1,rt+1,…,st+n)\tau = (s_t, a_t, r_t, s_{t+1}, a_{t+1}, r_{t+1}, \ldots, s_{t+n})τ=(st​,at​,rt​,st+1​,at+1​,rt+1​,…,st+n​).

These sequences help characterize two types of problems we will deal with. The first case happens when all sequences of actions from any sss are guaranteed to eventually reach some terminal state sTs_TsT​. We call these environments _episodic_ , or _finite-horizon_ , because interacting with them for long enough is guaranteed to eventually form an episode. The second case happens when there are trajectories that will never end. That is, sequences of timesteps that never reach some sTs_TsT​. We call these environments _non-episodic_ or _infinite-horizon_ because interacting with them forever is not guaranteed to form an episode. 

This distinction is important when we consider a _return_ Gj∈RG_j \in \mathbb{R}Gj​∈R, which is the cumulative reward obtained from timestep jjj onward along a trajectory. There is one return per timestep.

In the simplest episodic case, the return from timestep jjj can be written as

Gj=∑t=jT−1rt.G_j = \sum_{t=j}^{T-1} r_t.Gj​=t=j∑T−1​rt​.

For non-episodic or infinite-horizon settings, we introduce a _discount factor_ γ∈[0,1]\gamma \in [0, 1]γ∈[0,1] and define the _discounted return from timestep_ jjj as

Gj=∑t=j∞γ t−j rtG_j = \sum_{t=j}^{\infty} \gamma^{\,t-j} \, r_tGj​=t=j∑∞​γt−jrt​

Note that for finite-horizon episodic tasks, setting γ=1\gamma = 1γ=1 recovers the undiscounted return. 

The discount factor serves two purposes. First, it ensures that the infinite-horizon return converges so long as 0≤γ<10 \leq \gamma < 10≤γ<1 by forming a convergent geometric series when ∣rt∣|{r_t}|∣rt​∣ is bounded. Second, it acts as a form of temporal _credit assignment_ by assigning more weight to rewards that were obtained closer to the current time ttt. 

### 1.2 Value Functions​

The _state value function_ , often just called the _value function_ V:S→RV : \mathcal{S} \rightarrow \mathbb{R}V:S→R is a function that maps states to the _expected return_ of a policy at that state. It is given by 

V(st)=Eπ[Gt∣st].V(s_t) = \mathbb{E}_{\pi}[G_t \mid s_t].V(st​)=Eπ​[Gt​∣st​].

This is an important quantity to understand because it captures the _quality_ of a policy at a given state. It should be emphasized that the value function considers only one specific policy, so every time we make even a tiny change to our agent's policy, the value function will change as well. One way to envision the value function is to imagine the agent being dropped into the game at some arbitrary state sss. The _value_ of the policy at that state is the return it would get _on average_ if we allowed it to play from that state infinitely many times, restarting from the same state each time a terminal state is reached.

The _state-action value function_ , or _Q function_ Q:S×A→RQ : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}Q:S×A→R is a function that maps states and actions to the _expected return_ of a policy at a state sss when the agent takes action aaa at that state, then acts according to π\piπ forever afterwards. It is given by 

Q(st,at)=Eπ[Gt∣st,at].Q(s_t, a_t) = \mathbb{E}_{\pi}[G_t \mid s_t, a_t].Q(st​,at​)=Eπ​[Gt​∣st​,at​].

This quantity is similar to V(s)V(s)V(s), but with the caveat that the agent must first take the action ata_tat​ at state sts_tst​ before acting according to the policy π\piπ forever afterwards. Note that, in general, we can write Q(s,a)Q(s, a)Q(s,a) in terms of V(s)V(s)V(s) as

Q(st,at)=E[ rt+γ V(st+1)∣st,at ].Q(s_t, a_t) = \mathbb{E}[\, r_t + \gamma \, V(s_{t+1}) \mid s_t, a_t \,].Q(st​,at​)=E[rt​+γV(st+1​)∣st​,at​].

The _state-action advantage function_ , or _advantage function_ A:S×A→RA : \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}A:S×A→R, is the difference between the Q function and the value function at a state given an action. This is given by 

A(st,at)=Q(st,at)−V(st).A(s_t, a_t) = Q(s_t, a_t) - V(s_t).A(st​,at​)=Q(st​,at​)−V(st​).

Think of the advantage function as a measure of how much better it was to take the action ata_tat​ at state sts_tst​ than it would have been to just act according to the policy π\piπ at that state.

## 2\. The Learning Process​

This section will outline the general process of learning a policy from a set of trajectories. The derivation of the policy gradient shown here comes from OpenAI's [Spinning Up blog](<https://spinningup.openai.com/en/latest/spinningup>).

Most learning algorithms optimize an _objective_ J:Π→RJ: \Pi \rightarrow \mathbb{R}J:Π→R; for π∈Π\pi \in \Piπ∈Π, J(π)=Eπ[G0]J(\pi) = \mathbb{E}_{\pi}[G_0]J(π)=Eπ​[G0​] (where G0G_0G0​ denotes the return starting at time 000). The goal of learning is then to find a policy π∗\pi^*π∗ that maximizes the objective function, i.e. J(π∗)=max⁡π∈ΠJ(π)J(\pi^*) = \max_{\pi \in \Pi} J(\pi)J(π∗)=maxπ∈Π​J(π). For our purposes we care about the objective J(π)=Eπ[G0]J(\pi) = \mathbb{E}_{\pi}[G_0]J(π)=Eπ​[G0​]. However, there are many options.

The most common way to maximize this function is a process called _gradient ascent_ (though you may have heard of gradient _descent_ , the concepts are the same), which is an iterative process by which a policy is repeatedly adjusted in the direction of the gradient of the objective function ∇J(π)\nabla J(\pi)∇J(π). To do this in practice we will only consider policies that are differentiable functions parameterized by θ∈Rd\theta \in \mathbb{R}^dθ∈Rd, which we will write as πθ\pi_{\theta}πθ​.

### 2.1 The Policy Gradient​

This formulation of the policy and objective allows us to write an update rule for any parameters θu\theta_uθu​ by gradient ascent on JJJ, 

θu+1=θu+η∇θuJ(πθu).\theta_{u+1} = \theta_u + \eta \nabla_{\theta_{u}} J(\pi_{\theta_u}).θu+1​=θu​+η∇θu​​J(πθu​​).

where 0<η<∞0 < \eta < \infty0<η<∞ is referred to as the _learning rate_.

To derive ∇θJ(πθ)\nabla_{\theta} J(\pi_{\theta})∇θ​J(πθ​) we will first rewrite our objective Eπθ[G0]\mathbb{E}_{\pi_{\theta}}[G_0]Eπθ​​[G0​] in integral form as 

Eπθ[G0]=∫τP(τ∣πθ) G0(τ) dτ.\mathbb{E}_{\pi_{\theta}}[G_0] = \int_{\tau} P(\tau \mid \pi_{\theta}) \, G_{0}(\tau) \, d\tau.Eπθ​​[G0​]=∫τ​P(τ∣πθ​)G0​(τ)dτ.

Note that to evaluate this integral we need to know P(τ∣πθ)P(\tau \mid \pi_{\theta})P(τ∣πθ​), which is the probability of a trajectory occurring under πθ\pi_{\theta}πθ​. For a trajectory of length TTT this is given by

P(τ∣πθ)=P(s0)∏t=0T−1[ πθ(at∣st) P(st+1∣st,at) ].P(\tau \mid \pi_{\theta}) = P(s_0) \prod_{t=0}^{T-1} \big[\, \pi_{\theta}(a_t \mid s_t) \, P(s_{t+1} \mid s_t, a_t) \,\big].P(τ∣πθ​)=P(s0​)t=0∏T−1​[πθ​(at​∣st​)P(st+1​∣st​,at​)].

That is, the probability of a trajectory given a policy is the product of the probabilities of each state transition and the probabilities of each action taken over that trajectory. This would be cumbersome to differentiate on its own, but because P(τ∣πθ)P(\tau \mid \pi_{\theta})P(τ∣πθ​) specifies a probability function, we can employ the [log-derivative trick](<https://andrewcharlesjones.github.io/journal/log-derivative.html>) to get

∇θP(τ∣πθ)=P(τ∣πθ)∇θlog⁡P(τ∣πθ).\nabla_{\theta} P(\tau \mid \pi_{\theta}) = P(\tau \mid \pi_{\theta}) \nabla_{\theta} \log P(\tau \mid \pi_{\theta}).∇θ​P(τ∣πθ​)=P(τ∣πθ​)∇θ​logP(τ∣πθ​).

This may not seem useful at first, but log⁡P(τ∣πθ)\log P(\tau \mid \pi_{\theta})logP(τ∣πθ​) separates that nasty product from earlier into a summation:

log⁡P(τ∣πθ)=log⁡P(s0)+∑t=0T−1[ log⁡πθ(at∣st)+log⁡P(st+1∣st,at) ].\log P(\tau \mid \pi_{\theta}) = \log P(s_0) + \sum_{t=0}^{T-1} \big[\, \log \pi_{\theta}(a_t \mid s_t) + \log P(s_{t+1} \mid s_t, a_t) \,\big].logP(τ∣πθ​)=logP(s0​)+t=0∑T−1​[logπθ​(at​∣st​)+logP(st+1​∣st​,at​)].

This means we can differentiate each term of log⁡P(τ∣πθ)\log P(\tau \mid \pi_{\theta})logP(τ∣πθ​) with respect to θ\thetaθ to get its gradient, and neither P(s0)P(s_0)P(s0​) or P(st+1∣st,at)P(s_{t+1} \mid s_t, a_t)P(st+1​∣st​,at​) are functions of θ\thetaθ, so their contribution to ∇θlog⁡P(τ∣πθ)\nabla_{\theta} \log P(\tau \mid \pi_{\theta})∇θ​logP(τ∣πθ​) is zero. Therefore, we can write its gradient as

∇θlog⁡P(τ∣πθ)=∑t=0T−1∇θlog⁡πθ(at∣st).\nabla_{\theta} \log P(\tau \mid \pi_{\theta}) = \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t).∇θ​logP(τ∣πθ​)=t=0∑T−1​∇θ​logπθ​(at​∣st​).

Circling back to our goal of finding ∇θP(τ∣πθ)\nabla_{\theta} P(\tau \mid \pi_{\theta})∇θ​P(τ∣πθ​), we have 

∇θP(τ∣πθ)=P(τ∣πθ)∑t=0T−1∇θlog⁡πθ(at∣st).\nabla_{\theta} P(\tau \mid \pi_{\theta}) = P(\tau \mid \pi_{\theta}) \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t).∇θ​P(τ∣πθ​)=P(τ∣πθ​)t=0∑T−1​∇θ​logπθ​(at​∣st​).

Finally we can plug this back into the integral form of our objective as above to get

∇θJ(πθ)=∫τ∇θP(τ∣πθ) G0(τ) dτ.=∫τP(τ∣πθ) ∇θlog⁡P(τ∣πθ) G0(τ) dτ.=Eπθ[ ∑t=0T−1∇θlog⁡πθ(at∣st) Gt ].\begin{aligned} \nabla_{\theta} J(\pi_{\theta}) &= \int_{\tau} \nabla_{\theta} P(\tau \mid \pi_{\theta}) \, G_{0}(\tau) \, d\tau. \\\ &= \int_{\tau} P(\tau \mid \pi_{\theta}) \, \nabla_{\theta} \log P(\tau \mid \pi_{\theta}) \, G_{0}(\tau) \, d\tau. \\\ &= \mathbb{E}_{\pi_{\theta}}\Big[\, \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \, G_t \,\Big]. \end{aligned}∇θ​J(πθ​)​=∫τ​∇θ​P(τ∣πθ​)G0​(τ)dτ.=∫τ​P(τ∣πθ​)∇θ​logP(τ∣πθ​)G0​(τ)dτ.=Eπθ​​[t=0∑T−1​∇θ​logπθ​(at​∣st​)Gt​].​

Putting this together, we can approximate ∇θJ(πθ)\nabla_{\theta} J(\pi_{\theta})∇θ​J(πθ​) over a group of BBB timesteps as

∇θJ(πθ)≈1B∑t=1B∇θlog⁡πθ(at∣st)Gt.\nabla_{\theta} J(\pi_{\theta}) \approx \frac{1}{B} \sum_{t=1}^{B} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) G_t.∇θ​J(πθ​)≈B1​t=1∑B​∇θ​logπθ​(at​∣st​)Gt​.

Note that BBB is typically referred to as the _batch size_. Larger batch sizes will yield more accurate gradient estimates, but will also require more memory and time to compute.

One interesting property of this gradient is that it requires the probability of any action in a batch according to the policy to be non-zero (log⁡0\log 0log0 is undefined) and also not equal to one (log⁡1=0\log 1 = 0log1=0). For this estimator of the policy gradient, the policy should be stochastic so that log⁡π(at∣st)\log \pi(a_t \mid s_t)logπ(at​∣st​) is well-defined and informative. Deterministic policies can be handled by different estimators.

### 2.2 Baselines and The Critic​

When approximating ∇θJ(πθ)\nabla_{\theta} J(\pi_{\theta})∇θ​J(πθ​) as we derived above, we might run into a problem in settings where returns have a high variance. We would rather find a method to estimate ∇θJ(πθ)\nabla_{\theta} J(\pi_{\theta})∇θ​J(πθ​) that is less sensitive to the variance of returns. To do this, we will introduce a _baseline_ b(s)b(s)b(s), which is any function of the state (not dependent on the action), into our approximation:

∇θJ(πθ)≈1B∑t=1B∇θlog⁡πθ(at∣st)(Gt−b(st)).\nabla_{\theta} J(\pi_{\theta}) \approx \frac{1}{B} \sum_{t=1}^{B} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) (G_t - b(s_t)).∇θ​J(πθ​)≈B1​t=1∑B​∇θ​logπθ​(at​∣st​)(Gt​−b(st​)).

A comprehensive analysis of b(s)b(s)b(s) and what it should be is beyond the scope of this article, but the key concept to understand is that because b(s)b(s)b(s) only depends on the state, it does not change the expectation of our gradient estimate at all:

Eπθ[ ∇θlog⁡πθ(at∣st) b(st)]=0∴Eπθ[ ∇θlog⁡πθ(at∣st) (Gt−b(st))]=Eπθ[∇θlog⁡πθ(at∣st) Gt].\begin{aligned} \mathbb{E}_{\pi_{\theta}}\Big[ \, \nabla_{\theta} \log \pi_{\theta} (a_t \mid s_t) \, b(s_t) \Big] &= 0 \\\ \therefore \mathbb{E}_{\pi_{\theta}}\Big[ \, \nabla_{\theta} \log \pi_{\theta} (a_t \mid s_t) \, (G_t - b(s_t)) \Big] &= \mathbb{E}_{\pi_{\theta}}\Big[\nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t) \, G_t \Big]. \end{aligned}Eπθ​​[∇θ​logπθ​(at​∣st​)b(st​)]∴Eπθ​​[∇θ​logπθ​(at​∣st​)(Gt​−b(st​))]​=0=Eπθ​​[∇θ​logπθ​(at​∣st​)Gt​].​

However, b(s)b(s)b(s) does change the variance of the gradient estimator. It turns out that a near-optimal choice of baseline (and the one we will use going forward) is V(s)V(s)V(s). One way we can intuit the usefulness of this choice is to think of GtG_tGt​ as how the agent actually performed at the state sts_tst​, and V(st)V(s_t)V(st​) as how the agent was expected to perform at that state, so we are weighting the gradient estimate such that it points away from actions that led to below-average performance and towards actions that led to above-average performance.

Now that we are equipped with a baseline, we need to figure out how to compute it. Since V(s)V(s)V(s) is an expectation, we could try to approximate it by visiting every sts_tst​ many times and taking the average of the returns we get, but this is obviously impractical for environments with more than a few states. Instead, we will parameterize a function to approximate V(s)V(s)V(s) and optimize it by [least squares regression](<https://en.wikipedia.org/wiki/Least_squares>).

Thankfully, learning a model of V(s)V(s)V(s) is much easier than learning π\piπ. We will denote our model vϕ(s)v_{\phi}(s)vϕ​(s) with parameters ϕ∈Rc\phi \in \mathbb{R}^cϕ∈Rc. Our objective is then to learn ϕ∗\phi^*ϕ∗ such that vϕ∗(st)=V(st)v_{\phi^*}(s_t) = V(s_t)vϕ∗​(st​)=V(st​). Because V(s)V(s)V(s) is an expectation of returns according to π\piπ, we can consider each GtG_tGt​ a Monte Carlo sample from the distribution of returns under π\piπ. Therefore, we can approximate ϕ∗\phi^*ϕ∗ by performing gradient descent on 

Jϕ(st)=E[(Gt−vϕ(st))2].J_{\phi}(s_t) = \mathbb{E}[(G_t - v_{\phi}(s_t))^2].Jϕ​(st​)=E[(Gt​−vϕ​(st​))2].

### 2.3 Value Targets and TD(λ\lambdaλ)​

While what we have done so far is a valid way to compute value targets, we will encounter some issues in practice. Right now to train our critic we need to traverse a full trajectory in order to compute even one GtG_tGt​. In episodic environments with long episodes this can be impractical, and it is clearly impossible to do in non-episodic environments. To deal with these problems we need to introduce the famous [Bellman equation](<https://en.wikipedia.org/wiki/Bellman_equation>), which describes a recursive relationship between V(st)V(s_t)V(st​) and V(st+1)V(s_{t+1})V(st+1​). The Bellman equation rewrites the value function as follows:

V(st)=Eπ[Gt∣st]=Eπ[rt+γGt+1∣st]=Eπ[rt]+γV(st+1).\begin{aligned} V(s_t) &= \mathbb{E}_{\pi}[G_t \mid s_t] \\\ &= \mathbb{E}_{\pi}[r_t + \gamma G_{t+1} \mid s_t] \\\ &= \mathbb{E}_{\pi}[r_t] + \gamma V(s_{t+1}). \end{aligned}V(st​)​=Eπ​[Gt​∣st​]=Eπ​[rt​+γGt+1​∣st​]=Eπ​[rt​]+γV(st+1​).​

Which, so long as the reward function is deterministic, is equal to

V(st)=rt+γV(st+1).V(s_t) = r_t + \gamma V(s_{t+1}).V(st​)=rt​+γV(st+1​).

This is the Bellman equation. As you can see, this creates a recursive definition of the value function, where we can compute the value of one state as the sum of the reward at the current state and the value of the next. Further, we can write similar equations for V(st)V(s_t)V(st​) with up to as many rtr_trt​ terms as there are timesteps in a trajectory. All of the following are equal:

V(st)=rt+γV(st+1)=rt+γrt+1+γ2V(st+2)=rt+γrt+1+γ2rt+2+γ3V(st+3).⋮\begin{aligned} V(s_t) &= r_t + \gamma V(s_{t+1}) \\\ &= r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) \\\ &= r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V(s_{t+3}). \\\ &\vdots \end{aligned}V(st​)​=rt​+γV(st+1​)=rt​+γrt+1​+γ2V(st+2​)=rt​+γrt+1​+γ2rt+2​+γ3V(st+3​).⋮​

When we chose the value function as our baseline earlier, we approximated it by least squares regression on the returns GtG_tGt​. However, because all of these expansions are equal, it would be just as valid to train our critic by regression on any of these expansions of the Bellman equation. To make our lives easier let us rephrase the learning objective for the critic more generally as

Jϕ(st)=Eπ[(v^t−vϕ(st))2]J_{\phi}(s_t) = \mathbb{E}_{\pi}[ (\hat{v}_t - v_{\phi}(s_t))^2 ]Jϕ​(st​)=Eπ​[(v^t​−vϕ​(st​))2]

where v^t\hat{v}_tv^t​ is the learning target for vϕ(st)v_{\phi}(s_t)vϕ​(st​), which we previously set to v^t=Gt\hat{v}_t = G_tv^t​=Gt​. Let us now rewrite the above equalities and replace each V(s)V(s)V(s) with vϕ(st)v_{\phi}(s_t)vϕ​(st​). For convenience, we will label each expansion of the Bellman equation as Vt(n)V^{(n)}_tVt(n)​ where nnn is the number of rtr_trt​ terms in the calculation, like so:

Vt(1)=rt+γvϕ(st+1)Vt(2)=rt+γrt+1+γ2vϕ(st+2)Vt(3)=rt+γrt+1+γ2rt+2+γ3vϕ(st+3).⋮\begin{aligned} V^{(1)}_t &= r_t + \gamma v_{\phi}(s_{t+1}) \\\ V^{(2)}_t &= r_t + \gamma r_{t+1} + \gamma^2 v_{\phi}(s_{t+2}) \\\ V^{(3)}_t &= r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 v_{\phi}(s_{t+3}). \\\ &\vdots \end{aligned}Vt(1)​Vt(2)​Vt(3)​​=rt​+γvϕ​(st+1​)=rt​+γrt+1​+γ2vϕ​(st+2​)=rt​+γrt+1​+γ2rt+2​+γ3vϕ​(st+3​).⋮​

The question we are now faced with is which Vt(n)V^{(n)}_tVt(n)​ to choose as a replacement for GtG_tGt​ in our critic target. We could choose v^t=Vt(1)\hat{v}_t = V^{(1)}_tv^t​=Vt(1)​, or v^t=Vt(2)\hat{v}_t = V^{(2)}_tv^t​=Vt(2)​, or v^t=Vt(3)\hat{v}_t = V^{(3)}_tv^t​=Vt(3)​, etc, but the key insight behind TD(λ\lambdaλ) is that we can take an average over all Vt(n)V^{(n)}_tVt(n)​ rather than just choose one:

vt^=1n∑i=1nVt(i).\hat{v_t} = \frac{1}{n} \sum_{i=1}^n V^{(i)}_t.vt​^​=n1​i=1∑n​Vt(i)​.

This would work fine, but another important aspect to consider is the variance in each expansion Vt(n)V^{(n)}_tVt(n)​. Because there is variance in the returns, Vt(n)V^{(n)}_tVt(n)​ with fewer rtr_trt​ terms will have less variance. However, this also means vϕ(st)v_{\phi}(s_t)vϕ​(st​) has a larger influence over those Vt(n)V^{(n)}_tVt(n)​, and the critic is an imperfect estimator of V(s)V(s)V(s), so there will be a higher bias in those Vt(n)V^{(n)}_tVt(n)​. These facts outline a trade-off between variance and bias in our value targets; the more rtr_trt​ terms we use, the higher the variance, but the lower the bias. TD(λ\lambdaλ) addresses this trade-off by choosing a weighted average of all Vt(n)V^{(n)}_tVt(n)​ expansions, where the weight of each term is given by λn−1\lambda^{n-1}λn−1.

vt^=(1−λ)(Vt(1)+λVt(2)+λ2Vt(3)+λ3Vt(4)+…).\hat{v_t} = (1 - \lambda) (V^{(1)}_t + \lambda V^{(2)}_t + \lambda^2 V^{(3)}_t + \lambda^3 V^{(4)}_t + \ldots).vt​^​=(1−λ)(Vt(1)​+λVt(2)​+λ2Vt(3)​+λ3Vt(4)​+…).

Choosing vt^\hat{v_t}vt​^​ in this fashion gives us a single parameterized method of estimating V(st)V(s_t)V(st​) from a single trajectory that encompasses all the ways we might expand the Bellman equation, where λ∈[0,1)\lambda \in [0, 1)λ∈[0,1) controls the relative amounts of bias and variance in the learning targets.
