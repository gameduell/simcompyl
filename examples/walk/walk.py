"""Small example showing the features of the simulation framework."""

import simcompyl as sim
import numpy as np


class Walk(sim.Model):
    """The Walk model defines the bases for a random walk simulation.

    It take care of initialzing the samples x and y coordinatesand as well as
    calling the `walk` step periodically. This step uses a random distribution
    to get coordinate updates.
    """

    @sim.step
    def init(self):
        """
        Initialize the x and y coordinates of the samples.

        State
        -----
        x : float
            x coordinate of samples
        y : float
            y coordinate of samples
        """
        _init = super().init
        # bind indieces that can access specific parts of the state
        x = self.state(x=float)
        y = self.state(y=float)

        def impl(params, state):
            _init(params, state)
            state[x] = 0
            state[y] = 0

        return impl

    @sim.step
    def iterate(self):
        """Call the walk method to do the stepping.

        Steps
        -----
        walk(params, state) -> dx, dy
            step performing the actual walk
        """
        # get a binding to a sub-step
        _iterate = super().iterate
        _walk = self.walk

        def iterate(params, state):
            _iterate(params, state)

            # substeps also can return values or take extra arguments
            dx, dy = _walk(params, state)
            assert dx is not None
            assert dx is not None

        return iterate

    @sim.step
    def walk(self):
        """Simulate a step of a walk, returning the step.

        State
        -----
        x : float
            x coordinate of samples
        y : float
            y coordinate of samples

        Random
        ------
        step_x : float
            distirbution for changes in x direction
        step_y : float
            distirbution for changes in y direction

        Returns
        -------
        dx : float
            change in x direction
        dy : float
            change in y direction
        """
        # you also can retreive bindings as namedtuples
        idx = self.state(x=float, y=float)

        # here, we get bindings to generate distributions of random variables
        rnd = self.random(step_x=float, step_y=float)

        def impl(params, state):
            dx = rnd.step_x(params)
            dy = rnd.step_y(params)

            state[idx.x] += dx
            state[idx.y] += dy
            return dx, dy

        return impl


class Spawn(Walk):
    """Extends the basic walk with random spawn points."""

    @sim.step
    def init(self):
        """Randomize starting points of the samples.

        State
        -----
        x : float
            x coordinate of samples
        y : float
            y coordinate of samples

        Random
        ------
        initial_x : float
            distribution for initial x coordinates of samples
        initial_y : float
            distribution for initial y coordinates of samples
        """
        # again, we need to access the position
        x, y = self.state(x=float, y=float)

        # add some further requirements for the simulation
        rx, ry = self.random(initial_x=float, initial_y=float)

        # get binding to a step method of the super() object
        _init = super().init

        def impl(params, state):
            # so now we can refere to the super implementation
            _init(params, state)

            # but also have some custom behaviour
            state[x] = rx(params)
            state[y] = ry(params)

        return impl


class Energy(Walk):
    """Extends the basic walk model tracking and restriction samples energy."""

    @sim.step
    def init(self):
        """Initialize each sample with a random engery value.

        State
        -----
        energy : float
            energy remaining for the sample

        Random
        ------
        initial_energy : float
            distirbution for initial energy of samples
        """
        # additional bindings, also extending the state
        e = self.state(energy=float)
        ie = self.random(initial_energy=float)

        # bind to super method
        _init = super().init

        def impl(params, state):
            _init(params, state)
            state[e] = ie(params)

        return impl

    @sim.step
    def walk(self):
        """Trace and restrict energy of samples.

        If the sample still has energy left, a normal step is performed,
        taking energy according to the distance of the step. If energy is below
        zero, nothing is done.

        State
        -----
        energy : ...
            energy of the sample

        Retruns
        -------
        dx : float
            change in x direction or 0 if energy is below zero
        dy : float
            change in y direction or 0 if energy is below zero

        """
        # if we know that the type already is defined, we don't need to specify
        e = self.state(energy=...)

        # bind to super method
        _walk = super().walk

        def impl(params, state):
            if state[e] < 0:
                return .0, .0

            # call to super method with returned values used here
            dx, dy = _walk(params, state)
            state[e] -= np.sqrt(dx * dx + dy * dy)
            return dx, dy

        return impl


class ComplexWalk(Energy, Spawn):
    """Composition of Energy ans Spawn extension for the random Walk."""

    pass


class Simulation(sim.Allocation):
    """Simulation parameters for a random walk model."""

    n_samples = sim.Param("Number of samples to simulate", 1000)
    n_steps = sim.Param("Count of steps that each individual takes", 100)


class BasicDistance(sim.Allocation):
    """Distance parameters for a random walk model."""

    initial_x = sim.Normal("initial x coordinate", loc=0, scale=1)
    initial_y = sim.Normal("initial y coordinate", loc=0, scale=1)

    initial_energy = sim.Exponential("Initial energy of individuals",
                                     scale=1000)
    step_x = sim.Continuous("step in x direction", -.1, .1)
    step_y = sim.Continuous("step in y direction", -.1, .1)
