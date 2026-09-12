import numpy as np


def compare_state_evolutions_and_simulations(
    all_states1,
    all_states2,
    simulations: np.ndarray,
    label1="Method 1",
    label2="Method 2",
):
    """
    Compare the evolution of concept state bounds between two different state evolutions.

    Args:
        all_states1: First list of numpy arrays with shape (2, n_concepts)
                   representing [lower_bounds, upper_bounds] at each iteration
        all_states2: Second list of numpy arrays with same structure as all_states1
        label1: Label for the first method (default: "Method 1")
        label2: Label for the second method (default: "Method 2")
    """
    import matplotlib.pyplot as plt
    import numpy as np

    all_states1 = np.array(all_states1)
    all_states2 = np.array(all_states2)

    # Ensure both state evolutions have the same number of concepts
    assert (
        all_states1[0].shape[1] == all_states2[0].shape[1]
    ), "Both state evolutions must have the same number of concepts"

    n_concepts = all_states1[0].shape[1]

    # Create a figure with subplots for each concept
    fig, axes = plt.subplots(n_concepts, 1, figsize=(14, 2 * n_concepts))
    if n_concepts == 1:
        axes = [axes]  # Make it iterable if only one concept

    # Plot each concept's evolution
    for i in range(n_concepts):
        # Method 1 data
        lower1 = [state[0, i] for state in all_states1]
        upper1 = [state[1, i] for state in all_states1]

        # Method 2 data
        lower2 = [state[0, i] for state in all_states2]
        upper2 = [state[1, i] for state in all_states2]

        x1 = range(len(all_states1))
        x2 = range(len(all_states2))
        x3 = range(len(simulations[0]))

        # Plot method 1
        axes[i].fill_between(
            x1, lower1, upper1, alpha=0.2, color="blue", label=f"{label1} Range"
        )
        # axes[i].plot(x1, lower1, 'b-', linewidth=1, label=f'{label1} Bounds')
        # axes[i].plot(x1, upper1, 'b-', linewidth=1)

        # Plot method 2
        axes[i].fill_between(
            x2, lower2, upper2, alpha=0.2, color="red", label=f"{label2} Range"
        )
        # axes[i].plot(x2, lower2, 'r--', linewidth=1, label=f'{label2} Bounds')
        # axes[i].plot(x2, upper2, 'r--', linewidth=1)

        for sim in range(len(simulations)):
            axes[i].plot(x3, simulations[sim, :, i].tolist(), "k-")

        axes[i].set_ylabel(f"C{i+1}")
        axes[i].grid(True, linestyle="--", alpha=0.6)

        # Add legend to the first subplot only
        if i == 0:
            axes[i].legend(loc="upper right", bbox_to_anchor=(1.2, 1.3))

        # Only show x label on bottom plot
        if i == n_concepts - 1:
            axes[i].set_xlabel("Iteration")
        else:
            axes[i].set_xticklabels([])

        axes[i].set_xticks(list(range(0, all_states1.shape[0], 5)))

    plt.tight_layout()
    plt.show()
