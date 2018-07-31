### Overview
Implementation of algorithms from [Reinforcement Learning: An Introduction (2nd Edition) by Richard Sutton and Andrew Barto](http://incompleteideas.net/book/bookdraft2018mar21.pdf) and [David Silver's Reinforcement Learning Course](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

Code is written in Python 3 and uses RL environments from [OpenAI Gym](https://gym.openai.com/). Advanced techniques use [PyTorch](https://www.tensorflow.org/) for neural network implementations.


### Table of Contents

- [Multi-armed Bandits](Bandits/)
    - [Greedy](Bandits/Greedy.ipynb)
    - [Round Robin](Bandits/Round%20Robin.ipynb)
    - [Epsilon Greedy](Bandits/Epsilon%20Greedy.ipynb)
    - [Optimistic Greedy](Bandits/Optimistic%20Greedy.ipynb)
    - [UCB](Bandits/UCB.ipynb)
    - [Thompson Beta](Bandits/Thompson%20Beta.ipynb)
- [MDPs and Bellman Equations](MDP/)
- [Dynamic Programming: Model-Based RL, Policy Iteration and Value Iteration](DP/)
- [Monte Carlo Model-Free Prediction & Control](MC/)


### Resources

- [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/bookdraft2018mar21.pdf)
- [David Silver's Reinforcement Learning Course (UCL, 2015)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

### License

MIT License

Copyright for portions are held by Denny Britz as part of https://github.com/dennybritz/reinforcement-learning abd other portions are held by Microsoft as part of https://www.edx.org/course/reinforcement-learning-explained-0 .

Copyright (c) 2016 Denny Britz Copyright (c) 2018 Microsoft

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
