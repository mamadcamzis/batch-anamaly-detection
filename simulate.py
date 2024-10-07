import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))


# Simulate ride data function
def simulate_ride_distances():
    logger.info("Simulating ride distances ...")
    ride_dists = np.concatenate(
        (
            10 * np.random.random(size=370),
            30 * np.random.random(size=10),
            10 * np.random.random(size=10),
            10 * np.random.random(size=10)
        )
    )
    return ride_dists


def simulate_ride_data():
    logger.info("Simulating ride data ...")
    ride_dists = simulate_ride_distances()
    ride_speeds = simulate_ride_speed()
    ride_times = ride_dists / ride_speeds
    df = pd.DataFrame(
        {
        "ride_dists": ride_dists,
        "ride_times": ride_times,
        "ride_speeds": ride_speeds
        }
    )
    ride_ids = datetime.now().strftime("%Y%m%d") + df.index.astype(str)
    df["ride_ids"] = ride_ids
    return df


def simulate_ride_speed():
    logger.info("Simulating ride speeds ...")
    ride_speeds = np.concatenate(
        (
            np.random.normal(loc=30, scale=5, size=370),
            np.random.normal(loc=30, scale=5, size=10),
            np.random.normal(loc=50, scale=10, size=10),
            np.random.normal(loc=15, scale=4, size=10)
        )
    )
    return ride_speeds


if __name__ == "__main__":
    # simulate_ride_distances()
    # simulate_ride_speed()
    simulate_ride_data()
