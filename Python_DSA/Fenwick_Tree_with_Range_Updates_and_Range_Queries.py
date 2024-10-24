class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)
        self.lazy = [0] * (size + 1)  # For range updates

    def _update(self, index, value):
        while index <= self.size:
            self.tree[index] += value
            index += index & -index

    def _query(self, index):
        sum_ = 0
        while index > 0:
            sum_ += self.tree[index]
            index -= index & -index
        return sum_

    def update_range(self, left, right, value):
        # Range update using two point updates
        self._update(left, value)
        self._update(right + 1, -value)

    def query_range(self, index):
        return self._query(index)

# Example usage:
n = 10
fenwick = FenwickTree(n)

# Add 5 to the range [2, 6]
fenwick.update_range(2, 6, 5)

# Query the sum at index 4 (should reflect the range update)
print(f"Query at index 4: {fenwick.query_range(4)}")  # Output: 5

# Query the sum at index 7 (should not reflect the range update)
print(f"Query at index 7: {fenwick.query_range(7)}")  # Output: 0
