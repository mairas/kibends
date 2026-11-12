//! A union-find over indices, joining what touches into connected sets.

pub struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    pub fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
        }
    }

    pub fn root(&mut self, mut k: usize) -> usize {
        while self.parent[k] != k {
            self.parent[k] = self.parent[self.parent[k]];
            k = self.parent[k];
        }
        k
    }

    /// Joins the sets of `a` and `b`, under `b`'s root.
    pub fn join(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.root(a), self.root(b));
        self.parent[ra] = rb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joining_is_transitive() {
        let mut sets = UnionFind::new(4);
        sets.join(0, 1);
        sets.join(1, 2);
        assert_eq!(sets.root(0), sets.root(2));
        assert_ne!(sets.root(0), sets.root(3));
    }
}
