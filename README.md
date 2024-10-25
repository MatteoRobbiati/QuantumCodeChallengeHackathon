# Smart cities road resources optimization
## Team members: Matteo Robbiati, Andrea Papaluca, Andrea Pasquale, Simone Bordoni
### Quantum Code Challenge Hackathon for Smart Cities at CTE - Cagliari Digital Lab

## The idea

To date, it is possible to collect extensive data on cities. Of particular interest are data on the distribution of people across urban areas and traffic flows. Analyzing these data enables the optimization of emergency resources such as ambulances, fire departments, and patrol units. However, optimizing the allocation of these resources often involves solving computationally complex problems, which can become unsolvable in medium to large-sized cities. Additionally, because traffic flows can change rapidly, it is often necessary to continuously recalculate resource distribution, making computation time a critical factor, especially in emergency situations. One promising solution to this challenge involves the use of adiabatic quantum computers, which can quickly solve optimization problems.

## The algorithm

Our proposal is to frame the problem of road resource allocation as an instance of the well-known MaxCut problem. By dividing the city into regions, we construct a graph where each node represents a neighborhood or district, with edges reflecting the actual adjacency of these urban areas. The weight of each edge is assigned in direct proportion to the traffic intensity between the connected nodes. To estimate traffic levels, we analyze the population difference between nodes over two time intervals and apply a discrete derivative to approximate the traffic flow.
To optimize road resources and accelerate emergency response, it’s essential to avoid travel along heavily trafficked routes. Therefore, we propose calculating the MaxCut for the city’s graph to split the neighborhoods into two classes that maximize the weight of connecting edges between them. This allows us to allocate road resources between the two classes, minimizing travel time within each.
