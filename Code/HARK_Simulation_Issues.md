# HARK Simulation System Issues: Why We Can't Use Standard IndShockConsumerType

## Overview
This document explains why we encountered issues using the standard `IndShockConsumerType` from HARK and had to fall back to the custom `AltIndShockConsumerType` for the tax analysis implementation.

## The New HARK Simulation System (initialize_sym, symulate, hystory)

### What We Attempted
We tried to use the new HARK simulation methods as recommended:
- `initialize_sym(stop_dead=False)` instead of `initialize_sim()`
- `symulate()` instead of `simulate()`
- `hystory` instead of `history`

### The Core Problem: track_vars Not Properly Initialized

The main issue we encountered was with the `track_vars` attribute not being properly recognized by the new simulation system:

```
KeyError: 'aLvl'
File "HARK/simulator.py", line 1041, in reset
    self.history[var] = np.empty((T, N), dtype=self.types[var])
```

### Technical Details

1. **Simulator Initialization**: The new HARK simulation system requires agents to have a `_simulator` object that tracks specific variables defined in `track_vars`.

2. **track_vars Setting**: While we could set `agent.track_vars = ['aLvl', 'pLvl', 'WeightFac']` on individual agents, the simulator's internal `types` dictionary wasn't being populated correctly.

3. **Variable Type Detection**: The simulator needs to know the data types of variables to be tracked, but this wasn't happening automatically for custom tracking variables like `WeightFac`.

### Why AltIndShockConsumerType Works

The `AltIndShockConsumerType` from the original codebase works because:

1. **Custom sim_one_period Method**: It overrides the core simulation routine with a simplified version that manually handles `WeightFac`:
   ```python
   self.state_now["WeightFac"] = self.PopGroFac ** (-self.t_age)
   ```

2. **Legacy Simulation Methods**: It uses the older, more stable simulation methods (`initialize_sim()`, `simulate()`, `history`) that don't have the same initialization requirements.

3. **Proven Compatibility**: It was specifically designed to work with the existing parameter structure and simulation requirements.

## HARK Version Compatibility Issues

### Environment Setup
- **hetret environment**: HARK v0.16.0 - Missing `initialize_sym` method
- **hetret_taxes environment**: HARK master branch - Has new methods but incomplete implementation

### The Gap
Even with the master branch of HARK, the new simulation system appears to be:
1. **Incomplete**: The `initialize_sym` method exists but doesn't properly handle all tracking variables
2. **Documentation Gap**: The transition from old to new simulation methods isn't fully documented
3. **Backward Compatibility**: The new system doesn't seamlessly replace the old one for complex use cases

## Conclusion

For production use with complex heterogeneous agent models requiring custom tracking variables, the new HARK simulation system (`initialize_sym`, `symulate`, `hystory`) is not yet ready. The `AltIndShockConsumerType` with legacy simulation methods remains the most reliable approach.

### Recommendations
1. **Short-term**: Continue using `AltIndShockConsumerType` with legacy simulation methods
2. **Medium-term**: Monitor HARK development for completion of the new simulation system
3. **Long-term**: Migrate to new simulation methods once they're fully implemented and documented

This is a common issue in rapidly evolving open-source projects where new features are introduced before full backward compatibility is established.
