import torch

class GaLoreProjector:
    """
    Projects gradients into a low-rank subspace and back.
    Follows Algorithm 1 from the GaLore paper.
    """
    def __init__(self, rank, update_proj_gap=200, scale=0.25, proj_type='std'):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.ortho_matrix = None  # P (or Q)
        self.step_count = 0

    def project(self, full_rank_grad, iter):
        """
        Project gradient G (m×n) to low-rank R = Pᵀ G Q.
        Returns projected gradient.
        """
        m, n = full_rank_grad.shape

        if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
            # Recompute projection basis via SVD
            self.ortho_matrix = self._get_orthogonal_matrix(full_rank_grad)

        if self.proj_type == 'std':
            # Project: R = Pᵀ G  (or G Q depending on shape)
            if m >= n:
                # Project columns: G Q
                low_rank_grad = full_rank_grad @ self.ortho_matrix
            else:
                # Project rows: Pᵀ G
                low_rank_grad = self.ortho_matrix.T @ full_rank_grad
        return low_rank_grad

    def project_back(self, low_rank_grad):
        """
        Project low-rank update back to full space: ΔW = P R Qᵀ.
        """
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = low_rank_grad @ self.ortho_matrix.T
            else:
                full_rank_grad = self.ortho_matrix @ low_rank_grad
        return full_rank_grad * self.scale

    def _get_orthogonal_matrix(self, grad):
        """Compute truncated SVD, return left or right singular vectors."""
        m, n = grad.shape
        if m >= n:
            # Return right singular vectors (n × r)
            U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
            return Vh[:self.rank, :].T  # n × r
        else:
            # Return left singular vectors (m × r)
            U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
            return U[:, :self.rank]  # m × r
