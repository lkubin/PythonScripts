""" Dynamic Time Warping

    For reference:
        http://www.springer.com/cda/content/document/cda_downloaddocument/9783540740476-c1.pdf?SGWID=0-0-45-452103-p173751818
"""

import numpy as np
import pylab as plt


def dtw_cost(x, y, distance):
    """ Computes the dtw cost matrix on metric `distance`;
        accepts x, y as np.array (Npoint, Ndimension),
        Npoint could differ between them
    """
    n = len(x)
    m = len(y)
    dtw_m = np.zeros((n, m))
    # distance function must broadcast
    dtw_m[:, 0] = np.cumsum(distance(x, y[0]))
    dtw_m[0, :] = np.cumsum(distance(x[0], y))
    for i in range(1, n):
        for j in range(1, m):
            dtw_m[i, j] = min(dtw_m[i - 1, j - 1], dtw_m[i, j - 1], dtw_m[i - 1, j])
            dtw_m[i, j] += distance(x[i], y[j])
    return dtw_m


def dtw(x, y, distance):
    """ Computes dtw optimal path matrix and optimal cost matrix
        The total dtw cost is given by cost_matrix.sum()

        Args:
            x (np.ndarray): first input waveform (shape: (Nsample,), (Nsample, Ndim))
            y (np.ndarray): second input waveform (shape: (Nsample,), (Nsample, Ndim))
            distance (callable): metric function, must be capable of:
                - distance(x[i], y[j]) -> scalar
                - distance(x, y[i]) or distance(y, x[i]) ->
                                        vector (of shape (len(x),), (len(y),) respectively)
    """
    cost_matrix = dtw_cost(x, y, distance)
    opt_path = np.zeros_like(cost_matrix, dtype=int)

    opt_path[-1, -1] = 1

    n = len(x) - 1
    m = len(y) - 1

    while n > 0 and m > 0:
        distances = np.array(
            [cost_matrix[n - 1, m - 1], cost_matrix[n - 1, m], cost_matrix[n, m - 1]]
        )

        min_pos = np.argmin(distances)

        if min_pos == 0:
            opt_path[n - 1, m - 1] = 1
            m -= 1
            n -= 1
        elif min_pos == 1:
            opt_path[n - 1, m] = 1
            n -= 1
        else:
            opt_path[n, m - 1] = 1
            m -= 1

    if m > 0:
        opt_path[0, :m] = 1
    elif n > 0:
        opt_path[:n, 0] = 1
    cost_matrix[opt_path == 0] = 0.0
    return opt_path, cost_matrix


def wave_averager(x, y, distance=lambda x, y: np.abs(x - y)):
    """ Return the average of 2 monodimensional waveforms x, y possibly of different
        duration. DTW expansions and contractions are cropped before averaging
    """
    i, j = 0, 0
    logical_matrix, _ = dtw(x, y, distance)
    logical_matrix = logical_matrix > 0
    out_x, out_y = list(), list()

    while i < len(x) and j < len(y):
        x_row = logical_matrix[i, :]
        y_row = logical_matrix[:, j]
        out_x.append(x[y_row].mean())
        out_y.append(y[x_row].mean())
        i += y_row.sum()
        j += x_row.sum()

    return (np.array(out_x) + np.array(out_y)) * 0.5


def test_main():
    stime = np.arange(0., 10., 0.05)
    f_base = np.cos(stime)
    f_base1 = np.r_[f_base[:50], [f_base[50]] * 20, f_base[50:]]
    f_base2 = np.r_[f_base1[:10], [f_base1[10]] * 20, f_base1[10:]]
    f_base += np.random.normal(scale=0.1, size=len(stime))
    f_base1 += np.random.normal(scale=0.1, size=len(f_base1))
    f_base2 += np.random.normal(scale=0.1, size=len(f_base2))

    gg, cost = dtw(f_base, f_base1, distance=lambda x, y: np.abs(x - y))

    plt.figure("DTW path")
    plt.imshow(gg, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.figure("DTW path-cost")
    plt.imshow(cost, aspect='auto', interpolation='nearest')
    plt.colorbar()
    plt.figure("base waveforms")
    plt.plot(f_base)
    plt.plot(f_base1)
    plt.plot(f_base2)

    m1 = wave_averager(f_base, f_base1)
    m2 = wave_averager(m1, f_base2)
    m3 = wave_averager(m1, m2)
    plt.figure("Average waveforms")
    plt.plot(m1, label='m1')
    plt.plot(m2, label='m2')
    plt.plot(m3, label='m3')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_main()
