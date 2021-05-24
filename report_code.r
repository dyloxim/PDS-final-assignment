## ------------------------------------------------------- ##
## complete source code for this document can be found at: ##
## https://github.com/joelstrouts/PDS-final-assignment     ##
## ------------------------------------------------------- ##

set.seed(20)
do_re_eval = FALSE

## ===================== ##
## SECTION 1: Blood data ##
## ===================== ##

## ------------------------- ##
## SUBSECTION: preprocessing ##
## ------------------------- ##

models <- list(c("nb", "nb"), c("lda", "lda"),
               c("regLogistic", "lr"), c("rf", "rf"))

## Reading Data
blood <- list()
blood[["dfs"]] <- list()

blood$dfs[["normal"]] <-
    read.table(std$get_fpath("normals.txt"), header = TRUE)
blood$dfs[["carrier"]] <-
    read.table(std$get_fpath("carriers.txt"), header = TRUE)

## Combine data frames into single df with categorical variable distinguishing
## between types
blood[["meta"]] <- list()
blood$meta[["idvs"]] <- c("m1", "m2", "m3", "m4")

blood$dfs$normal$carrier <- rep("normal", nrow(blood$dfs$normal))
blood$dfs$carrier$carrier <- rep("carrier", nrow(blood$dfs$carrier))

blood$dfs$main <- rbind(blood$dfs$normal, blood$dfs$carrier)

blood$dfs$main$date <- str_pad(blood$dfs$main$date, width=6, side="left", pad="0")
blood$dfs$main$date <- as.Date(sub("(..)(..)(..)", "19\\3-\\1-01", blood$dfs$main$date))
blood$dfs$main$month <-
    as.numeric(round(difftime(blood$dfs$main$date, "1978-01-01", unit="days")/30.42))

blood$dfs[["unscaled"]] <- blood$dfs$main
blood$dfs$main[blood$meta$idvs] <-
    scale(blood$dfs$main[blood$meta$idvs])

blood$meta[["age_max"]] <-
    max(blood$dfs$main[which(blood$dfs$main$carrier == "normal"),"age"])
blood$meta[["mixed_diagnosis_ages"]] <-
    which(blood$dfs$main$age <= blood$meta$age_max)

## END-SECTION: preprocessing ##
## -------------------------- ##

## ------------------ ##
## SUBSECTION: Models ##
## ------------------ ##

## partition data
blood$meta[["train_idxs"]] <- createDataPartition(blood$dfs$main$carrier, p=0.7, list=FALSE)
blood$dfs[["train"]] <- blood$dfs$main[blood$meta$train_idxs,]
blood$dfs[["test"]] <- blood$dfs$main[-blood$meta$train_idxs,]

## train models
blood[["mdls"]] <- list()
blood$mdls[["pca-unscaled"]] <- prcomp(blood$dfs$unscaled[blood$meta$idvs])
blood$mdls[["pca-scaled"]] <- prcomp(blood$dfs$main[blood$meta$idvs])

# single feature classification
if(do_re_eval) {

    for(model in models) {
        blood$mdls[[model[[2]]]] <-
            std$multi_train(blood$dfs$train, blood$dfs$test,
                            blood$meta$idvs, "carrier",
                            model[[1]], trainControl(method = "LOOCV"))
    }
    store_blood_mdls <- blood$mdls
    save(store_blood_mdls, file='store/blood_mdls.Rda')

} else {

    load("store/blood_mdls.Rda")
    blood$mdls <- store_blood_mdls

}

blood$mdls[["ft_pairs"]] <- list()
blood$meta[["var_pairs"]] <- list(
    c("m1", "m2"), c("m1", "m3"), c("m1", "m4"),
    c("m2", "m3"), c("m2", "m4"), c("m3", "m4"))

# feature pair models
if(do_re_eval) {

    for(model in models) {
        blood$mdls$ft_pairs[[model[[2]]]] <-
            std$multi_train(
                    blood$dfs$train, blood$dfs$test, blood$meta$var_pairs,
                    "carrier", model[[1]], trainControl(method = "LOOCV"))
    }
    store_blood_mdls_ft_pairs <- blood$mdls$ft_pairs
    save(store_blood_mdls_ft_pairs, file='store/blood_mdls_ft_pairs.Rda')

} else {

    load("store/blood_mdls_ft_pairs.Rda")
    blood$mdls$ft_pairs <- store_blood_mdls_ft_pairs

}

## partition data
blood$dfs[["reduced"]] <- blood$dfs$main[blood$meta$mixed_diagnosis_ages,]
blood$meta[["train_idxs_red"]] <- createDataPartition(blood$dfs$reduced[,1], p=0.7, list=FALSE)

blood$dfs[["train_red"]] <- blood$dfs$reduced[blood$meta$train_idxs_red,]
blood$dfs[["test_red"]] <- blood$dfs$reduced[-blood$meta$train_idxs_red,]

blood$mdls[["red"]] <- list()

## single feature classification on reduced data
if(do_re_eval) {

    for(model in models) {
        blood$mdls$red[[model[[2]]]] <-
            std$multi_train(
                    blood$dfs$train_red, blood$dfs$test_red, blood$meta$idvs,
                    "carrier", model[[1]], trainControl(method = "LOOCV"))
    }
    store_blood_mdls_red <- blood$mdls$red
    save(store_blood_mdls_red, file='store/blood_mdls_red.Rda')

} else {

    load("store/blood_mdls_red.Rda")
    blood$mdls$red <- store_blood_mdls_red

}

blood$mdls$red[["ft_pairs"]] <- list()
# feature pair models on reducced data
if(do_re_eval) {

    for(model in models) {
        blood$mdls$red$ft_pairs[[model[[2]]]] <-
            std$multi_train(
                    blood$dfs$train_red, blood$dfs$test_red, blood$meta$var_pairs,
                    "carrier", model[[1]], trainControl(method = "LOOCV"))
    }
    store_blood_mdls_red_ft_pairs <- blood$mdls$red$ft_pairs
    save(store_blood_mdls_red_ft_pairs, file='store/blood_mdls_red_ft_pairs.Rda')

} else {

    load("store/blood_mdls_red_ft_pairs.Rda")
    blood$mdls$red$ft_pairs <- store_blood_mdls_red_ft_pairs

}

## END-SECTION: Models ##
## ------------------- ##

## ------------------- ##
## SUBSECTION:  tables ##
## ------------------- ##

blood[["tables"]] <- list()

blood$tables[["model_results_overview"]] <- function(model_name, model_set) {
    model <- model_set[[model_name]]
    first_var <- model$id_vars[[1]]
    # initialize table with just first column
    comparison_table <- data.frame(
        as.data.frame(model$results[[first_var]]$overall)[,1])
    rename_candidate <- names(comparison_table)[[1]]
    names(comparison_table)[
        names(comparison_table) == rename_candidate] <- first_var
    # rename rows
    new_row_names <- row.names(as.data.frame(model$results[[first_var]]$overall))
    new_row_names <- lapply(new_row_names,
               function(metric) {paste(toupper(model_name), metric, sep="_")})
    row.names(comparison_table) <- new_row_names
    # add other rows
    for(var in model$id_vars[-1]) {
        comparison_table[[var]] <-
            as.data.frame(model$results[[var]]$overall)[,1]
    }
    comparison_table
}

blood$tables[["compare_model_accuracies"]] <- function(model_names, model_set) {
    table <- blood$tables$model_results_overview(model_names[[1]], model_set)[1,]
    for(model_name in model_names[-1]) {
        table <- rbind(
            table,
            blood$tables$model_results_overview(model_name, model_set)[1,]
            )
    }
    table
}

## END-SECTION: tables ##
## ------------------- ##

## ------------------ ##
## SUBSECTION:  plots ##
## ------------------ ##

blood[["plots"]] <- list()

blood$plots[["class_histograms"]] <- function(variant) {
    layout(matrix(c(1,3,2,4,5,7,6,8), ncol=4, nrow=2))
    for(var in c("m1", "m2", "m3", "m4")) {
        for(class in c("carrier", "normal")) {
            if(class == "normal") {
                hist_col <- "blue"
            } else {
                hist_col <- "red"
            }
            hist_data <-
                blood$dfs$main[which(blood$dfs$main$carrier == class),var]
            if(variant == "auto-scale") {
                hist(hist_data, col=hist_col,
                     main=paste(var, class, sep = ", "),
                     xlab = paste(var, "(scaled)"))
            } else if(variant == "same-scale") {
                hist(hist_data, col=hist_col,
                     main=paste(var, class, sep = ", "),
                     xlab = paste(var, "(scaled)"),
                     xlim = c(-5,8), ylim=c(0,55))
            }
        }
    }
}

blood$plots[["class_qqnorms"]] <- function(variant) {
    layout(matrix(c(1,3,2,4,5,7,6,8), ncol=4, nrow=2))
    for(var in c("m1", "m2", "m3", "m4")) {
        for(class in c("normal", "carrier")) {
            if(class == "normal") {
                hist_col <- "blue"
            } else {
                hist_col <- "red"
            }
            hist_data <-
                blood$dfs$main[which(blood$dfs$main$carrier == class),var]
            qqnorm(hist_data, col=hist_col,
                 main=paste(var, class, sep = ", "), xlab = var)
        }
    }
}

blood$plots[["boxplots"]] <- function() {
    layout(matrix(1:4, nrow=2, ncol=2))
    boxplot(m1 ~ carrier, data=blood$dfs$main)
    boxplot(m2 ~ carrier, data=blood$dfs$main)
    boxplot(m3 ~ carrier, data=blood$dfs$main)
    boxplot(m4 ~ carrier, data=blood$dfs$main)
}

blood$plots[["pca"]] <- function() {
    layout(matrix(1:1, ncol=2, nrow=2))
    biplot(pca_blood)
    biplot(pca_blood_scaled)
    std$my_scatter__col_levels(
        pca_blood$x[,1], pca_blood$x[,2],
        color_by = blood$dfs$main$carrier,
        labels = TRUE,
        legend_pos = c(0.7,0.7))
    std$my_scatter__col_levels(
        pca_blood_scaled$x[,1], pca_blood_scaled$x[,2],
        color_by = blood$dfs$main$carrier,
        labels = TRUE,
        legend_pos = c(0.7,0))
}

blood$plots[["month_age_scatters"]] <- function() {
    layout(matrix(1:8, nrow=2, ncol=4))
    for(idv in c("age", "month")) {
        for(dv in c("m1","m2","m3","m4")) {
            std$my_scatter__col_levels(
                blood$dfs$main[,idv],
                blood$dfs$main[,dv],
                axis_labs = c(idv, dv),
                color_by = blood$dfs$main$carrier,
                colors = c("red", "blue"),
                plot_title = paste0(idv, " & ", dv))
            for(class in c("carrier", "normal")) {
                df <- blood$dfs$main[which(
                                  blood$dfs$main$carrier==class),]
                fit <- lm(
                    as.formula(paste0(
                        dv, "~poly(", idv, ",1)"
                        )),
                    data=df)
                idv_range <-
                    seq(0, 60, length.out = 140)
                newdat <- data.frame(idv = idv_range)
                names(newdat) <- c(idv)
                color <- ifelse((class == "carrier"), "red", "blue")
                pred <- predict(fit, newdata = newdat, interval = "confidence", level = 0.95)
                lines(x = idv_range, y = pred[,1], col = color, lwd = 1.7)
                lines(x = idv_range, y = pred[,2], col = color,
                      lty = "dashed")
                lines(x = idv_range, y = pred[,3], col = color,
                      lty = "dashed")
            }
        }
    }
}

blood$plots[["pair_scatters"]] <- function() {
    layout(matrix(c(1,4,2,5,3,6), ncol=3, nrow=2))
    for(pair in blood$meta$var_pairs) {
        std$my_scatter__col_levels(
                blood$dfs$main[,pair[[1]]], blood$dfs$main[,pair[[2]]],
                axis_labs = c(pair[[1]], pair[[2]]),
                plot_title = paste0(pair, collapse="_"),
                color_by = blood$dfs$main$carrier,
                shapes = c(1),
                colors = c("red", "blue"),
                legend_pos = c(0,0))
    }
}

## END-SECTION: plots ##
## ------------------ ##

## END SECTION 1         ##
## ===================== ##


## =================== ##
## SECTION 2: DNA data ##
## =================== ##

## -------------------------- ##
## SUBSECTION:  preprocessing ##
## -------------------------- ##

dna <- list()
dna[["dfs"]] <- list()

dna$dfs[["seqs"]] <- read.table(std$get_fpath("human-phage.txt"))
dna$dfs[["seqs_numeric"]] <- dna$dfs[["seqs"]]

# convert original table to numeric version
for(i in 2:101) {
    ## A->1, C->2 , G->3, T->4
    dna$dfs$seqs_numeric[,i] <-
        as.numeric(as.factor(dna$dfs$seqs[,i]))
}

dna$dfs[["ngrams"]] <- data.frame()
dna$meta[["the2grams"]] <- c()
dna$meta[["the3grams"]] <- c()

if(do_re_eval) {
    for(row in 1:length(dna$dfs$seqs[,1])) {
        for(win_size in 2:3) {
            for(col in 2:(101 - win_size)) {
                bases <- ""
                for(offset in 1:win_size) {
                    bases <- paste(bases, dna$dfs$seqs[row,col-1+offset], sep = "")
                }
                dna$dfs$ngrams[row,paste(col, win_size, sep = "_")] <- bases
                if(win_size == 2) {
                    dna$meta$the2grams <- c(dna$meta$the2grams, bases)
                } else if(win_size == 3) {
                    dna$meta$the3grams <- c(dna$meta$the3grams, bases)
                }
            }
        }
    }
    store_dna_ngrams <- list(
        ngrams_df = dna$dfs$ngrams,
        the2grams_v = dna$meta$the2grams,
        the3grams_v = dna$meta$the3grams
    )
    save(store_dna_ngrams, file="store/dna_ngrams.Rda")
} else {
    load("store/dna_ngrams.Rda")
    dna$dfs[["ngrams"]] <- store_dna_ngrams$ngrams_df
    dna$meta[["the2grams"]] <- store_dna_ngrams$the2grams_v
    dna$meta[["the3grams"]] <- store_dna_ngrams$the3grams_v
}

dna$dfs[["the2grams"]] <-
    data.frame(seq = unique(as.factor(dna$meta$the2grams)))
dna$dfs[["the3grams"]] <-
    data.frame(seq = unique(as.factor(dna$meta$the3grams)))

for(row in 1:length(dna$dfs$the2grams[,1])) {
    dna$dfs$the2grams[row,"count"] <-
        length(which(dna$meta$the2grams == dna$dfs$the2grams[row,"seq"]))
}

for(row in 1:length(dna$dfs$the3grams[,1])) {
    dna$dfs$the3grams[row,"count"] <-
        length(which(dna$meta$the3grams == dna$dfs$the3grams[row,"seq"]))
}

if(do_re_eval) {
    dna$dfs[["counts"]] <- data.frame(dna$dfs$seqs$V1)

    names(dna$dfs$counts)[names(dna$dfs$counts)=="dna.dfs.seqs.V1"] <- "class"

    for(row in 1:length(dna$dfs$seqs[,1])) {
        for(base in c("A", "C", "G", "T")) {
            longest_run <- 0
            current_run <- 0
            for(col in 1:length(dna$dfs$seqs[1,])) {
                if(dna$dfs$seqs[row, col] == base) {
                    current_run <- current_run + 1
                } else {
                    if(current_run > longest_run) {
                        longest_run <- current_run
                    }
                    current_run <- 0
                }
            }
            dna$dfs$counts[row, paste(base, "run", sep="_")] <- longest_run
        }
    }

    for(row in 1:length(dna$dfs$seqs[,1])) {
        print(paste("doing row", row))
        for(base in c("A", "C", "G", "T")) {
            dna$dfs$counts[row, paste0(base, "_count")] <-
                length(which(dna$dfs$seqs[row,2:101] == base))
        }
        for(seq in dna$dfs$the2grams[,"seq"]) {
            dna$dfs$counts[row, paste0(seq, "_count")] <-
                length(which(as.factor(dna$dfs$ngrams[row,]) == seq))
        }
        for(seq in dna$dfs$the3grams[,"seq"]) {
            dna$dfs$counts[row, paste0(seq, "_count")] <-
                length(which(as.factor(dna$dfs$ngrams[row,]) == seq))
        }
    }
    store_dna_counts <- dna$dfs$counts
    save(store_dna_counts, file="store/dna_counts.Rda")
} else {

    load("store/dna_counts.Rda")
    dna$dfs$counts <- store_dna_counts

}


for(i in 1:600) {
    if(dna$dfs$seqs_numeric[i,1] == "pos") {
        dna$dfs$seqs_numeric[i,1] <- "human"
        dna$dfs$seqs[i,1] <- "human"
        dna$dfs$counts[i,1] <- "human"
    } else if(dna$dfs$seqs_numeric[i,1] == "neg") {
        dna$dfs$seqs_numeric[i,1] <- "phage"
        dna$dfs$seqs[i,1] <- "phage"
        dna$dfs$counts[i,1] <- "phage"
    }
}

## END-SECTION: preprocessing ##
## -------------------------- ##

## ------------------- ##
## SUBSECTION:  Models ##
## ------------------- ##

dna[["mdls"]] <- list()

dna$meta[["train_idxs"]] <-
    createDataPartition(dna$dfs$seqs_numeric[,1], p=0.75, list=FALSE)

dna$dfs[["seqs_train"]] <- dna$dfs$seqs_numeric[dna$meta$train_idxs,]
dna$dfs[["seqs_test"]] <- dna$dfs$seqs_numeric[-dna$meta$train_idxs,]

dna$dfs[["counts_train"]] <- dna$dfs$counts[dna$meta$train_idxs,]
dna$dfs[["counts_test"]] <- dna$dfs$counts[-dna$meta$train_idxs,]

dna$mdls[["seqs_pca"]] <- prcomp(dna$dfs$seqs_numeric[,-1])
dna$mdls[["counts_pca"]] <- prcomp(dna$dfs$counts[,-1])

if(do_re_eval) {

    dna$mdls[["seqs_rf"]] <-
        std$train_model(
                dna$dfs$seqs_train, dna$dfs$seqs_test,
                -1, "V1", "rf", trainControl(method = "cv", number = 5), import = TRUE)

    dna$mdls[["counts_rf"]] <-
        std$train_model(
                dna$dfs$counts_train, dna$dfs$counts_test,
                -1, "class", "rf", trainControl(method = "cv", number = 5), import = TRUE)

    store_dna_mdls <- dna$mdls
    save(store_dna_mdls, file = "store/dna_mdls.Rda")

} else {
    load("store/dna_mdls.Rda")
    dna$mdls <- store_dna_mdls
}

## END-SECTION: Models ##
## ------------------- ##

## ------------------- ##
## SUBSECTION:  tables ##
## ------------------- ##

dna[["tables"]] <- list()

dna$tables[["rf_compare"]] <- function() {
    table <- rbind(
        dna$mdls$seqs_rf$results$overall,
        dna$mdls$counts_rf$results$overall
    )
    row.names(table) <- c("Unaltered", "Counts")
    names(table) <- c("Accuracy")
    data.frame(Accuracy = table[,c(1)])
}

## END-SECTION: tables ##
## ------------------- ##

## ------------------ ##
## SUBSECTION:  plots ##
## ------------------ ##
dna[["plots"]] <- list()

dna$plots[["pca_compare"]] <- function() {
    layout(matrix(1:4, nrow=2, ncol=2))
    biplot(dna$mdls$seqs_pca, main="Biplot for PCA on Unaltered Sequence Data")
    std$my_scatter__col_levels(
        dna$mdls$seqs_pca$x[,1], dna$mdls$seqs_pca$x[,2],
        color_by = dna$dfs$seqs[,1],
        colors = c("blue", "red"),
        axis_labs = c("PC1", "PC2"),
        plot_title = "Class Groupings Against PC1 & PC2 for Unaltered Sequence Data",
        legend_pos = c(0.8,0))
    biplot(dna$mdls$counts_pca, main="Biplot for PCA on Extracted Count Data")
    std$my_scatter__col_levels(
        dna$mdls$counts_pca$x[,1], dna$mdls$counts_pca$x[,2],
        color_by = dna$dfs$counts[,1],
        colors = c("blue", "red"),
        axis_labs = c("PC1", "PC2"),
        plot_title = "Class Groupings Against PC1 & PC2 For Extracted Count Data",
        legend_pos = c(0.7,0))
}

dna$plots[["rf_importance"]] <- function() {
    rf_var_imp <- varImp(dna$mdls$counts_rf$mdl)
    plot(rf_var_imp, top = 20)
}

## END-SECTION: plots ##
## ------------------ ##

initialised_workspace <- TRUE
