
library(volesti)
library(utils)

data_set_indices <- 0:7
value_function_types <- c('linear', 'concave', 'convex', 'concave&convex')
preferences_nums <- c(20, 40, 60, 80)
cognitive_biases <- c(0, 0.1, 0.2, 0.3)
random_indices <- 0:49

for (data_set_index in data_set_indices) {
    for (value_function_type in value_function_types) {
        for (preferences_num in preferences_nums) {
            for (cognitive_bias in cognitive_biases) {
                for (random_index in random_indices) {
                    # if (data_set_index==0 & value_function_type=='linear' & preferences_num==20 & cognitive_bias==0.3 & random_index==17){
                    # if (TRUE){
                    if (data_set_index==7 & value_function_type=="concave&convex"){
                    #     {}
                    # }
                    # else{
                        best_intervals_num_path = sprintf(f"{CODE_ROOT}/core/best_intervals_num/SGVIresults_bestintervalsnum_dataset%d_%s_%d_%f_random%d.txt", data_set_index, value_function_type, preferences_num, cognitive_bias, random_index)
                        best_intervals_num = as.numeric(readLines(best_intervals_num_path))
                        
                        i <- 0
                        repeat{
                            constr_path = sprintf(f"{CODE_ROOT}/core/constr_new/constr_dataset%d_%s_%d_%f_random%d_intervalsnum%d_index%d.csv", data_set_index, value_function_type, preferences_num, cognitive_bias, random_index, best_intervals_num, i)
                            
                            if (file.exists(constr_path)) {
                                print(constr_path)
                                # 读取CSV文件
                                data <- read.csv(constr_path, header = FALSE)
                                data_matrix <- as.matrix(data)

                                num_rows <- nrow(data_matrix)
                                num_cols <- ncol(data_matrix)
                                
                                # 最后一个单调性约束挪到倒数第一行，没有归一化约束了
                                data_matrix <- data_matrix[-num_cols, ]

                                A <- data_matrix[, 1:num_cols-1]
                                A[(num_cols):(num_rows-2), ] <- A[(num_cols):(num_rows-2), ] - data_matrix[(num_cols):(num_rows-2), num_cols]

                                
                                b <- -data_matrix[, num_cols]
                                b[(num_cols):(num_rows-2)] <- b[(num_cols):(num_rows-2)] + 10e-7
                                b[num_rows-1] <- 1

                                P = Hpolytope(A = A, b = b)
                                points =  sample_points(P, n = 10000, random_walk = list("walk" = "RDHR"))
                                
                                complete_points <- rbind(points, 1 - colSums(points))
                                # print(complete_points)
                                # print(colSums(complete_points))

                                # 指定要保存的文件名
                                samples_path <- sprintf(f"{CODE_ROOT}/core/samples_new/samples_dataset%d_%s_%d_%f_random%d_intervalsnum%d_index%d.csv", data_set_index, value_function_type, preferences_num, cognitive_bias, random_index, best_intervals_num, i)

                                
                                write.csv(complete_points, file = samples_path, row.names = FALSE, col.names = FALSE)

                                cat("Matrix saved to", samples_path, "\n")
                                print(dim(complete_points))

                                i <- i+1
                            }
                            else{
                                break
                            }
                        }
                    }
                    
                    
                }
            }
        }
    }
}

# A = matrix(c(-1,0,0,-1,1,1), ncol=2, nrow=3, byrow=TRUE)
# b = c(0,0,1)
# A = matrix(c(-1,1), ncol=1, nrow=2, byrow=TRUE)
# b = c(0,1)
# P = Hpolytope(A = A, b = b)
# points =  sample_points(P, n = 1000, random_walk = list("walk" = "RDHR"))
# # points = sample_points(P, n = 1000, distribution = list("density" = "gaussian", "variance" = 2))
# points1 <- as.vector(points)
# points2 <- 1-points1



# png("scatter_plot.png", width = 800, height = 600)
# # 绘制散点图
# plot(points1, points2, main = "Scatter Plot", xlab = "X-axis", ylab = "Y-axis", pch = 19, col = "blue")
# # 结束PNG绘图设备
# dev.off()