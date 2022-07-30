from verse.automaton.hybrid_automaton import HybridAutomaton

class HybridIoAutomaton(HybridAutomaton):
    def __init__(
        self,
        id = None, 
        input_variables = [],
        output_variables = [],
        internal_variables = [],
        discrete_variables = [],
        modes = [],
        edges = [],
        guards = [],
        resets = [],
        dynamics = {}
    ):
        super().__init__(
            id, 
            output_variables+internal_variables,
            discrete_variables,
            modes,
            edges,
            guards,
            resets,
            dynamics
        )
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.internal_variables = internal_variables

