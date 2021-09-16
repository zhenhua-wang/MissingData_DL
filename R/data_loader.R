library(tidyverse)

acs_downloader = function(year, save_dir) {
  base_url = "https://www2.census.gov/programs-surveys/acs/data/pums"
  type = "1-Year"
  
  url = str_c(base_url, year, type, sep = "/")
  save_location = str_c(save_dir, year, sep = "/")
  
  current_hname = str_c("csv_hus.zip", sep = "")
  current_pname = str_c("csv_pus.zip", sep = "")
  current_hurl = str_c(url, current_hname, sep = "/")
  current_purl = str_c(url, current_pname, sep = "/")
  download.file(url = current_hurl, destfile = str_c(save_location, current_hname, sep = "/"))
  download.file(url = current_purl, destfile = str_c(save_location, current_pname, sep = "/"))
  unzip(str_c(save_location, current_hname, sep = "/"),exdir=save_location)
  unzip(str_c(save_location, current_pname, sep = "/"),exdir=save_location)
}

acs_merge = function(year, fdir, issave = FALSE) {
  psam_husa <- read.csv(str_c(fdir, year, 'psam_husa.csv', sep = '/'))
  psam_husb <- read.csv(str_c(fdir, year, 'psam_husb.csv', sep = '/'))
  
  psam_pusa <- read.csv(str_c(fdir, year, 'psam_pusa.csv', sep = '/'))
  psam_pusb <- read.csv(str_c(fdir, year, 'psam_pusb.csv', sep = '/'))
  
  
  household <- rbind(psam_husa, psam_husb) %>%
    filter(TYPE == 1)
  personal <- rbind(psam_pusa, psam_pusb)%>%
    filter(RELP == 0)
  merged_df <- merge(household, personal, by = 'SERIALNO')
  
  if (issave) {
    write.csv(merged_df, str_c(save_dir, "house_person.csv", sep = "/"))
  }
  return(merged_df)
}