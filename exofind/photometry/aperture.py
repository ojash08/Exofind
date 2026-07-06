import numpy as np
import heapq


def build_aperture(image, start, threshold):

    rows, cols = image.shape

    mask = np.zeros_like(image, dtype=bool)
    growth_order = []

    heap = []

    r, c = start

    heapq.heappush(
        heap,
        (-image[r, c], r, c)
    )

    while heap:

        _, r, c = heapq.heappop(heap)

        if mask[r, c]:
            continue

        if image[r, c] < threshold:
            continue

        mask[r, c] = True
        growth_order.append((r, c))

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):

                if dr == 0 and dc == 0:
                    continue

                nr = r + dr
                nc = c + dc

                if 0 <= nr < rows and 0 <= nc < cols:
                    heapq.heappush(
                        heap,
                        (-image[nr, nc], nr, nc)
                    )

    return mask, growth_order