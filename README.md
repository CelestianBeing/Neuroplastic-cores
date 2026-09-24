# Neuroplastic SNN

A biologically inspired Spiking Neural Network (SNN) implemented in Python using **Leaky Integrate-and-Fire (LIF) neurons, Poisson spike encoding, and Spike-Timing-Dependent Plasticity (STDP)**.

The project takes text input, converts it into a numerical representation, transforms that representation into spike trains, processes the spikes through a small LIF neural network, and updates synaptic weights based on the timing of actual neuron spikes.

The current implementation is an **experimental prototype** focused on demonstrating neural dynamics and synaptic plasticity.

## Pros

* Implements a genuine **Leaky Integrate-and-Fire (LIF)** neuron model.
* Implements **Spike-Timing-Dependent Plasticity (STDP)** using actual spike timing.
* Converts numerical input into spike trains using **Poisson rate coding**.
* Uses real discrete-time neural dynamics rather than pre-generated results.
* Synaptic weights change based on the network's actual firing activity.
* Tracks neuron firing activity and synaptic weight changes during the simulation.
* Simple Python/NumPy implementation that is easy to inspect and modify.
* Provides a complete pipeline from **text input → neural spikes → SNN → synaptic learning**.
* Uses a fixed random seed option for reproducible experiments.
* Lightweight enough to run locally without specialized hardware.

## Cons

* The current implementation **does not use an actual LLM or transformer model**.
* The text representation is generated using character n-gram hashing rather than a semantic embedding model.
* The network is relatively small, using 32 input neurons and 16 output neurons in the included example.
* The current implementation does not perform a specific classification or prediction task.
* Poisson rate coding represents information primarily through spike frequency and may not capture all information contained in the original vector.
* The STDP implementation uses a relatively basic pair-based learning rule.
* The project does not currently demonstrate that the SNN performs better than conventional neural-network approaches.
* The system is a computational model inspired by biological neurons and should not be considered a complete biological simulation.
* The current output focuses mainly on spike counts and synaptic-weight changes rather than a practical end-user result.
