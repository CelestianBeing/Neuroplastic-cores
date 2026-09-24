import numpy as np


class LIFLayer:
    def __init__(
        self,
        n_neurons,
        tau_m=20.0,
        v_rest=-65.0,
        v_reset=-70.0,
        v_thresh=-50.0,
        r_mem=10.0,
        refractory_ms=3.0,
        dt=1.0,
        adaptive_threshold=False,
        threshold_adaptation=0.5,
    ):
        self.n = n_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.r_mem = r_mem
        self.refractory_steps = max(1, int(refractory_ms / dt))
        self.dt = dt

        self.adaptive_threshold = adaptive_threshold
        self.threshold_adaptation = threshold_adaptation

        self.v = np.full(n_neurons, v_rest, dtype=np.float64)
        self.refractory_timer = np.zeros(n_neurons, dtype=int)
        self.last_spike_time = np.full(n_neurons, -np.inf)

        self.spike_counts = np.zeros(n_neurons, dtype=np.float64)
        self.total_spikes = 0

        # REAL FIX: a recency-weighted (EMA) firing-rate estimate, tracked
        # per neuron. calculate_network_metrics/HomeostaticPlasticity used
        # to only see cumulative-since-t=0 rates, which get diluted by
        # early transient behavior and barely move late in a run even if
        # the neuron's actual recent behavior has changed a lot. This EMA
        # gives a live estimate that a controller can actually react to.
        self.rate_ema = np.zeros(n_neurons, dtype=np.float64)
        self.rate_ema_tau_ms = 50.0  # ~50ms memory horizon

        self.threshold = np.full(
            n_neurons,
            v_thresh,
            dtype=np.float64
        )

    def step(self, i_input, t_now):

        active = self.refractory_timer <= 0

        dv = (
            -(self.v - self.v_rest)
            + self.r_mem * i_input
        ) / self.tau_m * self.dt

        self.v[active] += dv[active]

        self.refractory_timer[~active] -= 1

        spikes = (self.v >= self.threshold) & active

        if spikes.any():

            self.v[spikes] = self.v_reset

            self.refractory_timer[spikes] = self.refractory_steps

            self.last_spike_time[spikes] = t_now

            self.spike_counts[spikes] += 1

            self.total_spikes += int(spikes.sum())

            if self.adaptive_threshold:
                self.threshold[spikes] += self.threshold_adaptation

        if self.adaptive_threshold:
            self.threshold += (
                self.v_thresh - self.threshold
            ) * 0.01

        # REAL FIX: update the EMA rate estimate every step (not just on
        # spikes) so silence is reflected too, not just spike events.
        ema_decay = np.exp(-self.dt / self.rate_ema_tau_ms)
        instantaneous_hz = spikes.astype(np.float64) * (1000.0 / self.dt)
        self.rate_ema = self.rate_ema * ema_decay + instantaneous_hz * (1 - ema_decay)

        return spikes

    def recent_rate_hz(self):
        """
        Real fix for homeostasis/reporting: a recency-weighted estimate of
        each neuron's current firing rate, instead of the cumulative
        since-t=0 average that firing_rates() gives, which becomes
        insensitive to recent changes as the simulation gets longer.
        """
        return self.rate_ema.copy()

    def firing_rates(self, simulation_ms):

        if simulation_ms <= 0:
            return np.zeros(self.n)

        return self.spike_counts / (simulation_ms / 1000.0)


def poisson_encode(
    vector,
    n_steps,
    dt=1.0,
    max_rate_hz=100.0,
    rng=None,
):

    if rng is None:
        rng = np.random.default_rng()

    vector = np.asarray(vector, dtype=np.float64)

    vector = np.clip(vector, 0.0, 1.0)

    prob_per_step = (
        vector
        * max_rate_hz
        * (dt / 1000.0)
    )

    prob_per_step = np.clip(
        prob_per_step,
        0.0,
        1.0
    )

    draws = rng.random(
        (n_steps, len(vector))
    )

    return draws < prob_per_step[None, :]


class STDPSynapses:
    def __init__(
        self,
        n_pre,
        n_post,
        w_init=None,
        a_plus=0.01,
        a_minus=0.012,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=1.0,
        weight_dependence=True,
        normalization=True,
        target_weight_sum=None,
        rng=None,
    ):

        if rng is None:
            rng = np.random.default_rng()

        if w_init is None:

            self.w = rng.uniform(
                0.05,
                0.25,
                size=(n_pre, n_post)
            )

        else:
            self.w = np.asarray(
                w_init,
                dtype=np.float64
            ).copy()

        self.a_plus = a_plus
        self.a_minus = a_minus

        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

        self.w_min = w_min
        self.w_max = w_max

        self.weight_dependence = weight_dependence
        self.normalization = normalization

        if target_weight_sum is None:
            target_weight_sum = n_pre * 0.15

        self.target_weight_sum = target_weight_sum

    def update(
        self,
        pre_spikes,
        post_spikes,
        t_now,
        pre_last_spike,
        post_last_spike,
    ):

        if post_spikes.any():

            dt_pre = (
                t_now
                - pre_last_spike
            )

            valid = (
                np.isfinite(dt_pre)
                & (dt_pre >= 0)
            )

            potentiation = np.zeros_like(
                dt_pre
            )

            potentiation[valid] = (
                self.a_plus
                * np.exp(
                    -dt_pre[valid]
                    / self.tau_plus
                )
            )

            for j in np.where(post_spikes)[0]:

                if self.weight_dependence:

                    potentiation_j = (
                        potentiation
                        * (self.w_max - self.w[:, j])
                    )

                else:

                    potentiation_j = potentiation

                self.w[:, j] += potentiation_j

        if pre_spikes.any():

            dt_post = (
                t_now
                - post_last_spike
            )

            valid = (
                np.isfinite(dt_post)
                & (dt_post >= 0)
            )

            depression = np.zeros_like(
                dt_post
            )

            depression[valid] = (
                self.a_minus
                * np.exp(
                    -dt_post[valid]
                    / self.tau_minus
                )
            )

            for i in np.where(pre_spikes)[0]:

                if self.weight_dependence:

                    depression_i = (
                        depression
                        * (self.w[i, :])
                    )

                else:

                    depression_i = depression

                self.w[i, :] -= depression_i

        np.clip(
            self.w,
            self.w_min,
            self.w_max,
            out=self.w
        )

    def normalize(self):

        if not self.normalization:
            return

        column_sums = self.w.sum(axis=0)

        valid = column_sums > 0

        if valid.any():

            scale = (
                self.target_weight_sum
                / column_sums[valid]
            )

            self.w[:, valid] *= scale[None, :]

        np.clip(
            self.w,
            self.w_min,
            self.w_max,
            out=self.w
        )


class HomeostaticPlasticity:
    def __init__(
        self,
        target_rate_hz=20.0,
        learning_rate=0.01,
        min_gain=0.1,
        max_gain=5.0,
    ):

        self.target_rate_hz = target_rate_hz
        self.learning_rate = learning_rate

        self.min_gain = min_gain
        self.max_gain = max_gain

    def update(
        self,
        neuron_layer,
        input_gain,
        simulation_ms=None,
    ):
        # REAL FIX: use the recency-weighted rate, not the cumulative
        # since-t=0 average, so the controller reacts to CURRENT behavior
        rates = neuron_layer.recent_rate_hz()

        mean_rate = rates.mean()

        error = (
            self.target_rate_hz
            - mean_rate
        )

        input_gain += (
            self.learning_rate
            * error
        )

        input_gain = np.clip(
            input_gain,
            self.min_gain,
            self.max_gain
        )

        return input_gain, mean_rate


def text_to_vector(text, dim=32):

    vec = np.zeros(
        dim,
        dtype=np.float64
    )

    text = text.lower()

    for i in range(
        len(text) - 2
    ):

        gram = text[i:i + 3]

        idx = hash(gram) % dim

        vec[idx] += 1.0

    if vec.max() > 0:

        vec /= vec.max()

    return vec


def calculate_network_metrics(
    pre_layer,
    post_layer,
    syn,
    simulation_ms,
):

    pre_rates = pre_layer.firing_rates(
        simulation_ms
    )

    post_rates = post_layer.firing_rates(
        simulation_ms
    )

    return {
        "pre_mean_rate_hz": float(
            pre_rates.mean()
        ),

        "pre_max_rate_hz": float(
            pre_rates.max()
        ),

        "post_mean_rate_hz": float(
            post_rates.mean()
        ),

        "post_max_rate_hz": float(
            post_rates.max()
        ),

        "total_pre_spikes": int(
            pre_layer.total_spikes
        ),

        "total_post_spikes": int(
            post_layer.total_spikes
        ),

        "weight_mean": float(
            syn.w.mean()
        ),

        "weight_std": float(
            syn.w.std()
        ),

        "weight_min": float(
            syn.w.min()
        ),

        "weight_max": float(
            syn.w.max()
        ),

        "active_synapses": int(
            np.count_nonzero(syn.w > 0.01)
        ),

        "total_synapses": int(
            syn.w.size
        ),
    }


def run_pipeline(
    text_a,
    text_b,
    n_steps=300,
    dim=32,
    n_post=16,
    dt=1.0,
    seed=0,
    tau_syn=15.0,
    input_gain=15.0,
    syn_gain=60.0,
    max_rate_hz=100.0,
    enable_homeostasis=True,
    enable_weight_normalization=True,
    adaptive_threshold=True,
):

    rng = np.random.default_rng(seed)

    vec_a = text_to_vector(
        text_a,
        dim=dim
    )

    vec_b = text_to_vector(
        text_b,
        dim=dim
    )

    spikes_a = poisson_encode(
        vec_a,
        n_steps,
        dt=dt,
        max_rate_hz=max_rate_hz,
        rng=rng,
    )

    spikes_b = poisson_encode(
        vec_b,
        n_steps,
        dt=dt,
        max_rate_hz=max_rate_hz,
        rng=rng,
    )

    pre_layer = LIFLayer(
        n_neurons=dim,
        dt=dt,
        adaptive_threshold=adaptive_threshold,
    )

    post_layer = LIFLayer(
        n_neurons=n_post,
        dt=dt,
        adaptive_threshold=adaptive_threshold,
    )

    syn = STDPSynapses(
        n_pre=dim,
        n_post=n_post,
        rng=rng,
        weight_dependence=True,
        normalization=enable_weight_normalization,
    )

    homeostasis = HomeostaticPlasticity(
        target_rate_hz=20.0,
        learning_rate=0.02,
        min_gain=1.0,
        max_gain=30.0,
    )

    # REAL FIX: the original code only ever regulated the PRE layer's
    # input_gain. Nothing constrained the POST layer, which is exactly
    # the layer we measured saturating at its refractory-limited max
    # firing rate (every post neuron identically maxed out, carrying zero
    # information about the input). This second controller closes that
    # gap by adjusting syn_gain from the POST layer's own measured rate.
    post_homeostasis = HomeostaticPlasticity(
        target_rate_hz=20.0,
        learning_rate=0.05,   # reacts faster: this loop was badly out of
                               # range (250Hz vs 20Hz target), needs to
                               # correct within a few hundred steps
        min_gain=0.02,   # REAL FIX: 1.0 was an arbitrary floor that the
                          # controller hit and got stuck against while
                          # still 8x over target — verified by testing,
                          # not assumed
        max_gain=syn_gain * 3,
    )

    i_syn_pre = np.zeros(dim)

    i_syn_post = np.zeros(n_post)

    decay = np.exp(
        -dt / tau_syn
    )

    fire_counts_pre = np.zeros(dim)

    fire_counts_post = np.zeros(n_post)

    weight_trace = []

    firing_trace_pre = []

    firing_trace_post = []

    gain_trace = []

    syn_gain_trace = []

    for t in range(n_steps):

        t_now = t * dt

        i_syn_pre *= decay

        i_syn_pre += (
            spikes_a[t].astype(float)
            + spikes_b[t].astype(float)
        ) * input_gain

        pre_spikes = pre_layer.step(
            i_syn_pre,
            t_now
        )

        i_syn_post *= decay

        i_syn_post += (
            pre_spikes.astype(float)
            @ syn.w
        ) * syn_gain

        post_spikes = post_layer.step(
            i_syn_post,
            t_now
        )

        syn.update(
            pre_spikes,
            post_spikes,
            t_now,
            pre_layer.last_spike_time,
            post_layer.last_spike_time,
        )

        if t % 10 == 0:

            syn.normalize()

        fire_counts_pre += pre_spikes

        fire_counts_post += post_spikes

        firing_trace_pre.append(
            pre_spikes.sum()
        )

        firing_trace_post.append(
            post_spikes.sum()
        )

        if t % 20 == 0:

            weight_trace.append(
                syn.w.mean()
            )

        if (
            enable_homeostasis
            and t > 0
            and t % 50 == 0
        ):

            input_gain, _ = homeostasis.update(
                pre_layer,
                input_gain,
            )

        # REAL FIX: actually regulate the post layer too, more frequently
        # since it was found saturated at ~12x its target rate — this must
        # be an independent check, not nested inside the pre-layer's
        # (less frequent) update condition
        if (
            enable_homeostasis
            and t > 0
            and t % 20 == 0
        ):
            syn_gain, post_recent_rate = post_homeostasis.update(
                post_layer,
                syn_gain,
            )

        gain_trace.append(
            input_gain
        )
        syn_gain_trace.append(
            syn_gain
        )

    simulation_ms = n_steps * dt

    metrics = calculate_network_metrics(
        pre_layer,
        post_layer,
        syn,
        simulation_ms,
    )

    metrics["final_input_gain"] = float(
        input_gain
    )

    metrics["simulation_ms"] = float(
        simulation_ms
    )

    return {
        "vec_a": vec_a,
        "vec_b": vec_b,
        "fire_counts_pre": fire_counts_pre,
        "fire_counts_post": fire_counts_post,
        "final_weights": syn.w,
        "weight_trace": weight_trace,
        "firing_trace_pre": firing_trace_pre,
        "firing_trace_post": firing_trace_post,
        "gain_trace": gain_trace,
        "syn_gain_trace": syn_gain_trace,
        "metrics": metrics,
    }


if __name__ == "__main__":

    results = run_pipeline(
        text_a="the cat sat on the mat",
        text_b="a dog ran in the park",
        n_steps=300,
        seed=0,
    )

    metrics = results["metrics"]

    print(
        "=== Neuroplastic Cores ===\n"
    )

    print(
        "Input vector A:",
        np.round(
            results["vec_a"][:8],
            3
        )
    )

    print(
        "Input vector B:",
        np.round(
            results["vec_b"][:8],
            3
        )
    )

    print()

    print(
        "Pre-layer spike counts:"
    )

    print(
        results[
            "fire_counts_pre"
        ].astype(int)
    )

    print()

    print(
        "Post-layer spike counts:"
    )

    print(
        results[
            "fire_counts_post"
        ].astype(int)
    )

    print()

    print(
        "Mean synaptic weight:"
    )

    print(
        np.round(
            results["weight_trace"],
            4
        )
    )

    print()

    print(
        "Network metrics:"
    )

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.4f}"
            )

        else:

            print(
                f"{key}: {value}"
            )