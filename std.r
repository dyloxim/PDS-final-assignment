suppressPackageStartupMessages(library("wrapr"))
suppressPackageStartupMessages(library("knitr"))
suppressPackageStartupMessages(library("kableExtra"))
suppressPackageStartupMessages(library("xtable"))
suppressPackageStartupMessages(library("stringr"))
suppressPackageStartupMessages(library("caret"))
suppressPackageStartupMessages(library("MASS"))
suppressPackageStartupMessages(library("randomForest"))
suppressPackageStartupMessages(library("LiblineaR"))

std <- list()

std[["xtable2kable"]] <- function(x) {
  out <- capture.output(print(x, table.placement = NULL))[-(1:2)]
  out <- paste(out, collapse = "\n")
  structure(out, format = "latex", class = "knitr_kable")
}

std[["get_fpath"]] <- 
    function (fname) { paste0("../resources/data-sets/", fname, collapse=",") }

std[["my_scatter__col_levels"]] <-
    function (xs, ys, ...,
              rows, color_by, shape_by,
              shapes = c(1), legend_pos = c(0,0),
              plot_title = "Plot", labels = FALSE,
              axis_labs = c("", ""),
              colors = c("red", "blue", "green")
              ) {
        ## abort if optional argument not specified above provided
        wrapr::stop_if_dot_args(substitute(list(...)), "my_scatter")

        if (missing(rows)) { rows <- 1:length(xs) }
        if (missing(shape_by)) { shape_by <- rep(1, length(xs)) }
        if (!missing(rows)) {
            xs <- xs[rows]
            ys <- ys[rows]
            shape_by <- shape_by[rows]
            color_by = color_by[rows]
        }

        plot(xs, ys,
             main=plot_title, xlab=axis_labs[1], ylab=axis_labs[2],
             cex.main=1.2, cex=1,
             col=colors[((as.numeric(as.factor(color_by)) - 1) %% length(colors)) + 1],
             pch=shapes[((as.numeric(as.factor(shape_by)) - 1) %% length(shapes)) + 1]
             )

        if (labels) {
            text(xs, ys, labels = 1:length(xs), cex=0.5, pos = 1)
        }

        ## c(0,0) := top left corner,
        ## c(1,1) := bottom right corner,
        legend_coords = c(
        (1 - legend_pos[1]) * min(xs) + legend_pos[1] * max(xs),
        legend_pos[2] * min(ys) + (1 - legend_pos[2]) * max(ys)
        )

        legend_values <- unique(as.factor(color_by))
        shape_values <- shapes[as.numeric(unique(as.factor(shape_by)))]
        col_values <- colors[as.numeric(unique(as.factor(color_by)))]

        legend(legend_coords[1], legend_coords[2], 
               legend=legend_values,
               cex=0.9,
               col = col_values,
               pch=shape_values)
    }


#' Scatter plot with 
std[["my_scatter__col_grad"]] <-
    function(xs, ys, ...,
             rows, color_by, color_label = "",
             shape_by, axis_labs = c("xs", "ys"),
             colors = c('black', 'red3'),
             shapes = c(15), legend_pos = c(0,0),
             title = "Plot"
             ) {
        ## abort if optional argument not specified above provided
        wrapr::stop_if_dot_args(substitute(list(...)), "my_scatter")

        if (missing(rows)) { rows <- 1:length(xs) }
        if (missing(shape_by)) { shape_by <- rep(1, length(xs)) }
        if (!missing(rows)) {
            xs <- xs[rows]
            ys <- ys[rows]
            shape_by <- shape_by[rows]
        }

        col_func <- colorRampPalette(colors)
        col_vals <- rep(1, length(xs))

        if (!missing(color_by)) {
            ## setup color palette
            col_vals <- as.numeric(color_by - min(color_by))
            if (max(col_vals) > 100) {
                interval <- max(col_vals)/100
                col_vals <- as.integer(round(col_vals/interval))
            }
            col_vals <- col_vals + 1
        }

        if (legend_pos[1] > 1) {
            par(xpd=TRUE, mar=c(4.5,4.5,1,6))
            legend_pos <- c(1.1,0)
        } else {
            par(xpd = FALSE)
        }

        plot(xs, ys,
             main=title, cex.main=1.2, cex=1.5,
             xlab=axis_labs[1], ylab=axis_labs[2],
             col=col_func(max(col_vals))[col_vals],
             pch=shapes[(as.numeric(as.factor(shape_by)) %% length(shapes)) + 1],
             )

        legend_coords = c(
        (1 - legend_pos[1]) * min(xs) + legend_pos[1] * max(xs),
        legend_pos[2] * min(ys) + (1 - legend_pos[2]) * max(ys)
        )


        if (!missing(color_by)) {
            ## draw legend for palette
            lgd_ = rep(NA, 11) 
            lgd_[c(1,6,11)] = c(
                min(color_by),
                round(mean(c(min(color_by),max(color_by)))),
                max(color_by))
            legend(x = legend_coords[1], y = legend_coords[2],
                   legend = lgd_,
                   fill = col_func(11),
                   border = NA,
                   y.intersp = 0.5,
                   title = color_label,
                   cex = 1, text.font = 1)
        }
    }

std[["multi_train"]] <- function(train, test,
                                 id_vars, d_var,
                                 model_name,
                                 train_control, ...,
                                 rows) {
    ## abort if optional argument not specified above provided
    wrapr::stop_if_dot_args(substitute(list(...)), "multi_train")

    varname <-
        function(id_var) { return(paste0(id_var, collapse = "_")) }

    output <- list()
    output[["models"]] <- list()
    output[["preds"]] <- list()
    output[["results"]] <- list()
    output[["id_vars"]] <- lapply(id_vars, varname)

    for (var_set in id_vars) {
        formula <- as.formula(paste0(d_var, "~", paste0(var_set, collapse="+")))
        output$models[[varname(var_set)]] <- train(form = formula,
                                           data = train,
                                           trControl = train_control,
                                           method = model_name)
    }

    for (id_var in id_vars) {
        output$preds[[varname(id_var)]] <-
            predict(output$models[[varname(id_var)]], test[,c(id_var, d_var)])
    }

    for (id_var in id_vars) {
        output$results[[varname(id_var)]] <- confusionMatrix(
            output$preds[[varname(id_var)]],
            as.factor(test[,d_var])
        )
    }
    output
}

std[["train_model"]] <- function(train, test,
                                 idvs, classes,
                                 train_method,
                                 train_control,
                                 import = FALSE)
{
    model <- list()

    model[["idvs"]] <- idvs

    if(idvs[[1]] == -1) {
        formula <- as.formula(paste0(classes, "~."))
    } else {
        formula <- as.formula(paste0(
            classes, "~", paste(idvs, collapse = "+", sep = ""))
            )
    }

    model[["mdl"]] <-
        train(form = formula,
              data = train,
              trControl = train_control,
              method = train_method,
              importance = import)

    model[["preds"]] <-
        predict(model$mdl, test)

    model[["results"]] <-
        confusionMatrix(
            model$preds,
            as.factor(test[,classes]))
    model
}
