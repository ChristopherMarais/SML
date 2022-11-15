x = c(73.8, 35.9, 42.3, 54.4, 74.5, 32.1)

knots = c(20, 40, 60)
x <- seq(min(knots)-1, max(knots)+1, length.out = 501)


bb <- splines::splineDesign(knots, x = x, ord=2, outer.ok = TRUE)

plot(range(x), c(0,1), type = "n", xlab = "x", ylab = "",
     main =  "B-splines - sum to 1 inside inner knots")
abline(v = knots, lty = 3, col = "light gray")
lines(x, rowSums(bb), col = "gray", lwd = 2)
matlines(x, bb, ylim = c(0,1), lty = 1)

