suppressPackageStartupMessages({
  library(matrixStats)
})

# Fast row-wise Pearson correlation between matrix rows and a vector
row_cor_pearson <- function(X, w) {
  stopifnot(ncol(X) == length(w))

  w0 <- as.numeric(scale(w))

  mu <- rowMeans(X, na.rm = TRUE)
  sd <- rowSds(X, na.rm = TRUE)
  sd[sd == 0] <- NA_real_

  Xz <- sweep(X, 1, mu, "-")
  Xz <- sweep(Xz, 1, sd, "/")

  as.numeric(Xz %*% w0) / (ncol(Xz) - 1)
}
