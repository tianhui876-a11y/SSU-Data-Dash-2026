# SSU Lasso: complete RStudio file-chooser version.
# Open this .R file in RStudio, then click Source.
# Choose CSV files in this exact order:
#   1. listings.csv
#   2. test_off_market.csv
#   3. test_list_aware.csv
# Files may live in DIFFERENT folders. Do not select Parquet files.
# The model/validation method matches SSU_Lasso_Final.R.

# STEP 1. Packages, three file pop-ups, and input checks.
packages <- c("glmnet", "Matrix")
missing_packages <- packages[!vapply(
  packages, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1)
)]
if (length(missing_packages)) {
  install.packages(missing_packages, repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(Matrix))
set.seed(311)
options(scipen = 999)

choose_csv <- function(step, expected_name) {
  cat("\n", step, ": choose ", expected_name, " in the pop-up.\n", sep = "")
  flush.console()
  path <- file.choose()
  if (tolower(tools::file_ext(path)) != "csv") {
    stop("Select a .csv file, not .parquet. Run the script again.")
  }
  normalizePath(path, mustWork = TRUE)
}

# POP-UP 1: Select listings.csv (150,000 labeled training homes).
listings_path <- choose_csv("POP-UP 1 of 3", "listings.csv")

# POP-UP 2: Select test_off_market.csv (3,974 homes without list price).
off_market_path <- choose_csv("POP-UP 2 of 3", "test_off_market.csv")

# POP-UP 3: Select test_list_aware.csv (4,064 homes with list price).
list_aware_path <- choose_csv("POP-UP 3 of 3", "test_list_aware.csv")

input_paths <- c(
  listings = listings_path,
  test_off_market = off_market_path,
  test_list_aware = list_aware_path
)
if (anyDuplicated(input_paths)) {
  stop("The same file was selected twice. Restart and choose three different CSVs.")
}

read_selected_csv <- function(path, expected_rows, required, forbidden) {
  d <- read.csv(path, stringsAsFactors = FALSE, encoding = "UTF-8")
  if (nrow(d) != expected_rows || !all(required %in% names(d)) ||
      any(forbidden %in% names(d))) {
    stop(paste("Wrong dataset selected:", basename(path),
               "Check the three selection steps and restart the script."))
  }
  for (nm in intersect(c("date_listed", "date_sold"), names(d))) {
    d[[nm]] <- as.Date(substr(d[[nm]], 1, 10))
  }
  stopifnot(!anyNA(d$ml_num), !anyDuplicated(d$ml_num),
            !anyNA(d$date_listed))
  d
}

sales <- read_selected_csv(
  listings_path, 150000,
  required = c("ml_num", "date_listed", "sold_price", "date_sold", "list_price"),
  forbidden = character()
)
tests <- list(
  off_market = read_selected_csv(
    off_market_path, 3974,
    required = c("ml_num", "date_listed", "house_type_name"),
    forbidden = c("sold_price", "list_price")
  ),
  list_aware = read_selected_csv(
    list_aware_path, 4064,
    required = c("ml_num", "date_listed", "house_type_name", "list_price"),
    forbidden = "sold_price"
  )
)
stopifnot(!anyNA(sales$date_sold), all(is.finite(sales$sold_price)),
          all(sales$sold_price > 0),
          all(sales$date_sold >= sales$date_listed),
          max(sales$date_sold) < as.Date("2026-06-01"))
for (d in tests) stopifnot(!any(d$ml_num %in% sales$ml_num))

cat("\nAll three datasets loaded correctly:\n")
print(c(training = nrow(sales),
        off_market = nrow(tests$off_market),
        list_aware = nrow(tests$list_aware)))

# Output goes beside the selected listings.csv, in a new timestamped folder.
# A fresh folder prevents accidental reuse of old model checkpoints.
out_dir <- file.path(
  dirname(listings_path),
  paste0("SSU_Lasso_Output_", format(Sys.time(), "%Y%m%d_%H%M%S"))
)
for (d in c(out_dir, file.path(out_dir, "figures"),
            file.path(out_dir, "models"), file.path(out_dir, "submissions"))) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}
if (!dir.exists(out_dir) || file.access(out_dir, 2) != 0) {
  stop("Cannot write beside listings.csv. Move it to a writable folder and restart.")
}
cat("\nResults folder:\n", normalizePath(out_dir), "\n")
write_out <- function(d, n) {
  write.csv(d, file.path(out_dir, n), row.names = FALSE)
}
write_out(data.frame(column = names(sales),
                    missing = colSums(is.na(sales))), "missingness.csv")

# STEP 2. Define only predictors available in each prediction task.
area_mid <- function(x) {
  # Vectorized row-local parsing. No prices or future rows enter this step.
  x<-gsub(",","",trimws(as.character(x)))
  x<-gsub("[\u2013\u2014]","-",x)
  out<-rep(NA_real_,length(x))
  exact<-!is.na(x) & grepl("^[0-9.]+$",x)
  closed<-!is.na(x) & grepl("^[0-9.]+\\s*-\\s*[0-9.]+$",x)
  out[exact]<-as.numeric(x[exact])
  lo<-as.numeric(sub("\\s*-.*$","",x[closed]))
  hi<-as.numeric(sub("^.*-\\s*","",x[closed]))
  out[closed]<-(lo+hi)/2
  out
}
make_x <- function(d,track,variant="full") {
  a<-area_mid(d$house_area); a[!is.finite(a)|a<=0]<-NA_real_
  lot<-d$lot_front*d$lot_depth;lot[!is.finite(lot)|lot<0]<-NA_real_
  m<-as.integer(format(d$date_listed,"%m"))
  x<-data.frame(bedrooms=d$bedrooms,washrooms=d$washrooms,
    parking_total=d$parking_total,garage=d$garage,
    log_area=log(a),log_lot=log1p(lot),
    area_missing=as.numeric(is.na(a)),lot_missing=as.numeric(is.na(lot)),
    latitude=d$latitude,longitude=d$longitude,
    time_years=as.numeric(d$date_listed-as.Date("2021-01-01"))/365.25,
    month_sin=sin(2*pi*m/12),month_cos=cos(2*pi*m/12),
    house_type_name=as.character(d$house_type_name),
    house_style=as.character(d$house_style),
    municipality=as.character(d$municipality),
    community=as.character(d$community))
  if(track=="list_aware") {
    stopifnot(all(is.finite(d$list_price)),all(d$list_price>0))
    x$log_list_price<-log(d$list_price)
  }
  drop<-switch(variant,full=character(),
    no_location=c("latitude","longitude","municipality","community"),
    no_size=c("log_area","log_lot","area_missing","lot_missing"),
    no_time=c("time_years","month_sin","month_cos"),
    no_list="log_list_price",stop("Unknown feature variant"))
  x[,setdiff(names(x),drop),drop=FALSE]
}

# STEP 3. Fit medians and dummy encoding on training rows only.
# This transparent base-R implementation avoids a recipes dependency.
encode <- function(x,pp) {
  blocks<-list()
  for(nm in pp$numeric) {
    z<-as.numeric(x[[nm]]);z[!is.finite(z)]<-pp$medians[[nm]]
    b<-Matrix(z,ncol=1,sparse=TRUE);colnames(b)<-nm
    blocks[[length(blocks)+1]]<-b
  }
  for(nm in pp$categorical) {
    z<-as.character(x[[nm]]);z[is.na(z)|trimws(z)==""]<-"__missing__"
    lev<-pp$levels[[nm]]
    # One reference level omitted. Unseen labels use the reference pattern.
    j<-match(z,lev)-1L; ok<-!is.na(j)&j>0L
    if(length(lev)>1) {
      b<-sparseMatrix(i=which(ok),j=j[ok],x=1,
        dims=c(nrow(x),length(lev)-1),
        dimnames=list(NULL,paste(nm,lev[-1],sep="=")))
      blocks[[length(blocks)+1]]<-b
    }
  }
  do.call(cbind,blocks)
}
fit_pp <- function(x) {
  pp<-list(numeric=names(x)[vapply(x,is.numeric,logical(1))],
           categorical=names(x)[vapply(x,is.character,logical(1))])
  pp$medians<-lapply(x[pp$numeric],function(z) {
    z<-z[is.finite(z)];if(!length(z)) stop("All-missing numeric feature")
    median(z)
  })
  pp$levels<-lapply(x[pp$categorical],function(z) {
    z[is.na(z)|trimws(z)==""]<-"__missing__";sort(unique(z))
  })
  xm<-encode(x,pp)
  vari<-colMeans(xm^2)-colMeans(xm)^2
  pp$keep<-colnames(xm)[vari>1e-12]
  pp
}
mat <- function(x,pp) encode(x,pp)[,pp$keep,drop=FALSE]
score <- function(y,p) {
  e<-100*abs(p-y)/y
  c(n=length(y),MdAPE=median(e),MAPE=mean(e),
    signed=median(100*(p-y)/y))
}
month_back <- function(d,n) seq(d,by=paste0("-",n," months"),length.out=2)[2]
window_rows <- function(d,cutoff,w) {
  if(is.infinite(w)) d else d[d$date_sold>=month_back(cutoff,w),]
}

# STEP 4. Select penalty and history length using inner historical data.
# Grid and history candidates fixed before looking at outer-month errors.
lambda_grid<-10^seq(-0.3,-5,length.out=50)
history_candidates<-c(Inf,24,12)
fit_lasso <- function(train,track,cutoff,variant="full",
                      windows=history_candidates) {
  tuning_start<-month_back(cutoff,3)
  older<-train[train$date_sold<tuning_start,]
  tune<-train[train$date_sold>=tuning_start,]
  stopifnot(nrow(older)>0,nrow(tune)>0)
  curves<-list();best_error<-Inf;best<-NULL
  for(w in windows) {
    tr<-window_rows(older,tuning_start,w)
    pp<-fit_pp(make_x(tr,track,variant))
    xm<-mat(make_x(tr,track,variant),pp)
    path<-glmnet(xm,log(tr$sold_price),family="gaussian",alpha=1,
                 lambda=lambda_grid,standardize=TRUE,thresh=1e-8)
    lp<-predict(path,mat(make_x(tune,track,variant),pp),s=lambda_grid)
    errs<-apply(exp(lp),2,function(p) score(tune$sold_price,p)[["MdAPE"]])
    curves[[length(curves)+1]]<-data.frame(history_months=w,
                          lambda=lambda_grid,MdAPE=errs)
    j<-which.min(errs)
    if(errs[j]<best_error) {
      best_error<-errs[j];best<-list(window=w,lambda=lambda_grid[j])
      # Earlier calibration residuals, not outer evaluation labels.
      best$q90<-as.numeric(quantile(abs(log(tune$sold_price)-lp[,j]),0.90))
    }
  }
  tr<-window_rows(train,cutoff,best$window)
  pp<-fit_pp(make_x(tr,track,variant));xm<-mat(make_x(tr,track,variant),pp)
  model<-glmnet(xm,log(tr$sold_price),alpha=1,family="gaussian",
    lambda=lambda_grid,standardize=TRUE,thresh=1e-8)
  stopifnot(model$jerr==0)
  list(pp=pp,model=model,track=track,variant=variant,lambda=best$lambda,
       history=best$window,q90=best$q90,n_train=nrow(tr),p=ncol(xm),
       tuning=do.call(rbind,curves),inner_MdAPE=best_error)
}
predict_lasso <- function(fit,d) {
  lp<-as.numeric(predict(fit$model,mat(make_x(d,fit$track,fit$variant),fit$pp),
                         s=fit$lambda))
  p<-round(exp(lp));stopifnot(all(is.finite(p)),all(p>0))
  data.frame(predicted=p,lower=exp(lp-fit$q90),upper=exp(lp+fit$q90))
}
coefficients <- function(fit) {
  b<-as.matrix(coef(fit$model,s=fit$lambda))
  data.frame(predictor=rownames(b),coefficient=as.numeric(b[,1]),
             selected=rownames(b)!="(Intercept)" & b[,1]!=0)
}
save_fit <- function(fit,stem) {
  write_out(coefficients(fit),paste0(stem,"_coefficients.csv"))
  write_out(fit$tuning,paste0(stem,"_tuning.csv"))
  png(file.path(out_dir,"figures",paste0(stem,"_tuning.png")),1000,650)
  cs<-split(fit$tuning,fit$tuning$history_months)
  plot(range(lambda_grid),range(fit$tuning$MdAPE),log="x",type="n",
       xlab="Lambda",ylab="Inner tuning MdAPE (%)")
  for(i in seq_along(cs)) lines(cs[[i]]$lambda,cs[[i]]$MdAPE,col=i,lwd=2)
  abline(v=fit$lambda,lty=2,col="grey40")
  legend("topleft",legend=paste(names(cs),"months"),col=seq_along(cs),lty=1)
  dev.off()
  png(file.path(out_dir,"figures",paste0(stem,"_paths.png")),1000,650)
  plot(fit$model,xvar="lambda",label=FALSE)
  abline(v=log(fit$lambda),lty=2);dev.off()
}

# STEP 5. Reproduce the spatial starter and a transparent local-median baseline.
# Baselines enforce sold_date < listing_date for each home.
baselines <- function(train,homes) {
  tr<-train[order(-as.numeric(train$date_sold),train$ml_num,method="radix"),]
  pools<-split(seq_len(nrow(tr)),tr$house_type_name)
  nearest<-local<-numeric(nrow(homes))
  for(i in seq_len(nrow(homes))) {
    h<-homes[i,];pool<-pools[[h$house_type_name]]
    pool<-pool[tr$date_sold[pool]<h$date_listed]
    if(!length(pool)) pool<-which(tr$date_sold<h$date_listed)
    stopifnot(length(pool)>0)
    dist<-((tr$longitude[pool]-h$longitude)*0.72)^2+
           (tr$latitude[pool]-h$latitude)^2
    take<-head(order(dist,method="radix"),10)
    nearest[i]<-median(tr$sold_price[pool[take]])
    # 365-day interpretation of the handout's previous 12 months.
    recent<-pool[tr$date_sold[pool]>=h$date_listed-365]
    nb<-recent[tr$community[recent]==h$community]
    local[i]<-median(tr$sold_price[if(length(nb)) nb else
                                  if(length(recent)) recent else pool])
  }
  data.frame(nearest=round(nearest),local_median=round(local),
             list_price=homes$list_price)
}

# STEP 6. Outer validation and ablations: never tune on these answers.
months<-seq(as.Date("2025-12-01"),as.Date("2026-05-01"),by="month")
variants<-c("full","no_location","no_size","no_time")
all_results<-all_coefs<-all_selections<-list();k<-q<-v<-1
for(i in seq_along(months)) {
  cutoff<-months[i];end<-seq(cutoff,by="month",length.out=2)[2]
  tr<-sales[sales$date_sold<cutoff,]
  homes<-sales[sales$date_sold>=cutoff & sales$date_sold<end,]
  mon<-format(cutoff,"%Y-%m")
  cat("MONTH",mon,"train",nrow(tr),"evaluate",nrow(homes),"\n")
  cache<-file.path(out_dir,paste0("checkpoint_",mon,".rds"))
  if(file.exists(cache)) {
    z<-readRDS(cache)
  } else {
    b<-baselines(tr,homes)
    z<-list(rows=list(),coefs=list(),settings=list());a<-c<-j<-1
    for(track in c("off_market","list_aware")) {
      settings<-c("guide_all_history",variants,
                   if(track=="list_aware") "no_list")
      for(setting in settings) {
        variant<-if(setting=="guide_all_history") "full" else setting
        fit<-fit_lasso(tr,track,cutoff,variant,
                 windows=if(setting=="guide_all_history") Inf else history_candidates)
        pred<-predict_lasso(fit,homes)
        z$rows[[a]]<-data.frame(ml_num=homes$ml_num,track=track,
          model=setting,month=mon,house_type=homes$house_type_name,
          community=homes$community,actual=homes$sold_price,
          predicted=pred$predicted,lower=pred$lower,upper=pred$upper)
        a<-a+1
        z$settings[[j]]<-data.frame(track=track,model=setting,month=mon,
          lambda=fit$lambda,history_months=fit$history,n_train=fit$n_train,
          candidate_columns=fit$p,selected=sum(coefficients(fit)$selected),
          inner_MdAPE=fit$inner_MdAPE)
        j<-j+1
        cat(track,setting,"MdAPE",round(score(homes$sold_price,pred$predicted)[2],3),
            "lambda",fit$lambda,"history",fit$history,"\n")
        if(setting=="full") {
          cf<-coefficients(fit);cf$track<-track;cf$month<-mon
          z$coefs[[c]]<-cf;c<-c+1
          save_fit(fit,paste(track,mon,sep="_"))
        }
      }
      bn<-if(track=="off_market") c("nearest","local_median") else c("list_price","local_median")
      for(nm in bn) {
        z$rows[[a]]<-data.frame(ml_num=homes$ml_num,track=track,model=nm,
          month=mon,house_type=homes$house_type_name,community=homes$community,
          actual=homes$sold_price,predicted=b[[nm]],lower=NA_real_,upper=NA_real_)
        a<-a+1
      }
    }
    saveRDS(z,cache)
  }
  all_results[[k]]<-do.call(rbind,z$rows);k<-k+1
  all_coefs[[q]]<-do.call(rbind,z$coefs);q<-q+1
  all_selections[[v]]<-do.call(rbind,z$settings);v<-v+1
}
val<-do.call(rbind,all_results);cf<-do.call(rbind,all_coefs)
settings<-do.call(rbind,all_selections)
write_out(val,"validation_predictions.csv")
write_out(cf,"validation_coefficients.csv")
write_out(settings,"selected_settings.csv")

# STEP 7. Answer the guide with measured results and limitations.
group_scores<-function(d,groups) {
  pieces<-split(d,interaction(d[groups],drop=TRUE,lex.order=TRUE))
  do.call(rbind,lapply(pieces,function(z)
    cbind(z[1,groups,drop=FALSE],as.data.frame(t(score(z$actual,z$predicted))))))
}
summary<-group_scores(val,c("track","model"));print(summary)
write_out(summary,"model_comparison.csv")
write_out(group_scores(val,c("track","model","month")),"monthly_errors.csv")
full<-val[val$model=="full",]
write_out(group_scores(full,c("track","house_type")),"errors_by_type.csv")
write_out(group_scores(full,c("track","community")),"errors_by_community.csv")
full$price_band<-cut(full$actual,c(0,5e5,1e6,1.5e6,Inf),
  labels=c("under_500k","500k_to_1m","1m_to_1.5m","over_1.5m"),right=FALSE)
write_out(group_scores(full,c("track","price_band")),"errors_by_price.csv")
cf<-cf[cf$predictor!="(Intercept)",]
stab<-aggregate(selected~track+predictor,cf,sum)
stab$selection_fraction<-stab$selected/length(months)
write_out(stab,"selection_stability.csv")
coverage<-do.call(rbind,lapply(split(full,full$track),function(z)
 data.frame(track=z$track[1],nominal=0.9,
   coverage=mean(z$actual>=z$lower & z$actual<=z$upper),
   median_width=median(z$upper-z$lower))))
write_out(coverage,"interval_coverage.csv")
# These historical residual bands are empirical, with no exchangeability guarantee.
set.seed(311)
ci<-do.call(rbind,lapply(split(full,full$track),function(z) {
  b<-split(z,z$month)
  draws<-replicate(1000,{
    ids<-sample(seq_along(b),length(b),replace=TRUE)
    zz<-do.call(rbind,b[ids]);score(zz$actual,zz$predicted)[["MdAPE"]]
  })
  data.frame(track=z$track[1],low=quantile(draws,.025),
             high=quantile(draws,.975),method="month-block bootstrap; only six months")
}))
write_out(ci,"mdape_uncertainty.csv")

# STEP 8. Refit and write all hidden-home predictions, preserving IDs.
final_settings<-list()
for(track in names(tests)) {
  fit<-fit_lasso(sales,track,as.Date("2026-06-01"))
  pred<-predict_lasso(fit,tests[[track]])
  out<-data.frame(ml_num=tests[[track]]$ml_num,predicted_price=pred$predicted)
  stopifnot(nrow(out)==nrow(tests[[track]]),!anyDuplicated(out$ml_num),
            identical(out$ml_num,tests[[track]]$ml_num),
            all(is.finite(out$predicted_price)),all(out$predicted_price>0))
  write_out(out,file.path("submissions",paste0("submission_",track,".csv")))
  write_out(cbind(ml_num=out$ml_num,pred),paste0(track,"_uncertainty.csv"))
  save_fit(fit,paste0(track,"_final"))
  saveRDS(fit,file.path(out_dir,"models",paste0(track,".rds")))
  final_settings[[track]]<-data.frame(track=track,lambda=fit$lambda,
    history_months=fit$history,n_train=fit$n_train,candidate_columns=fit$p,
    selected=sum(coefficients(fit)$selected),inner_MdAPE=fit$inner_MdAPE)
}
write_out(do.call(rbind,final_settings),"final_settings.csv")
writeLines(capture.output(sessionInfo()),file.path(out_dir,"sessionInfo.txt"))
# Hash the actual selected files, even when they live in different folders.
write_out(data.frame(
  dataset = names(input_paths),
  file = unname(input_paths),
  md5 = unname(tools::md5sum(input_paths))
), "input_md5.csv")
cat("COMPLETE: validation, coefficients, figures, models, two submissions.\n")
cat("\nYour results are saved here:\n", normalizePath(out_dir), "\n")
cat("Final CSV files are inside the submissions subfolder.\n")
