# app.R - Healthcare Data Analysis & Linear Regression Model
library(shiny)
library(shinydashboard)
library(tidyverse)
library(caret)
library(plotly)
library(DT)
library(pROC)
library(performance)
library(broom)
library(corrplot)

ui <- dashboardPage(
  dashboardHeader(title = "Healthcare Analytics Dashboard"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Data Generation", tabName = "data", icon = icon("database")),
      menuItem("Exploratory Analysis", tabName = "explore", icon = icon("chart-bar")),
      menuItem("Linear Regression", tabName = "regression", icon = icon("chart-line")),
      menuItem("Model Performance", tabName = "performance", icon = icon("tachometer-alt")),
      menuItem("Download Reports", tabName = "reports", icon = icon("download")),
      br(),
      shiny::tags$div(
        style = "padding: 15px;",
        h4("Model Configuration:"),
        sliderInput("train_split", "Train/Test Split (%):",
                    min = 60, max = 90, value = 70, step = 5),
        actionButton("generate_data", "Generate New Dataset", 
                     class = "btn-primary btn-block"),
        actionButton("train_model", "Train Model", 
                     class = "btn-success btn-block")
      )
    )
  ),
  
  dashboardBody(
    shiny::tags$head(
      shiny::tags$style(HTML("
        .small-box {height: 80px;}
        .info-box {min-height: 70px;}
        .metric-box {
          background-color: #f8f9fa;
          border: 1px solid #dee2e6;
          border-radius: 5px;
          padding: 10px;
          margin-bottom: 10px;
        }
        .highlight {
          background-color: #e7f3ff;
          border-left: 4px solid #007bff;
        }
      "))
    ),
    
    tabItems(
      # Tab 1: Data Generation
      tabItem(
        tabName = "data",
        fluidRow(
          column(12,
                 box(
                   title = "Healthcare Dataset Generator",
                   status = "primary",
                   solidHeader = TRUE,
                   width = 12,
                   h4("Simulated Healthcare Data for Linear Regression"),
                   p("Generating 300 data points with:"),
                   shiny::tags$ul(
                     shiny::tags$li(shiny::tags$b("Dependent Variable:"), "Patient Health Score (numeric, 0-100)"),
                     shiny::tags$li(shiny::tags$b("Independent Variables:")),
                     shiny::tags$ul(
                       shiny::tags$li("Age (numeric, 18-80)"),
                       shiny::tags$li("BMI (numeric, 18-40)"),
                       shiny::tags$li("Smoker Status (binary: 0=Non-smoker, 1=Smoker)"),
                       shiny::tags$li("Treatment Type (categorical with 4 levels: A, B, C, D)")
                     )
                   ),
                   hr(),
                   actionButton("simulate_data", "Simulate Dataset", 
                                class = "btn-primary btn-lg"),
                   br(), br(),
                   DTOutput("data_table")
                 )
          )
        ),
        fluidRow(
          column(6,
                 box(
                   title = "Dataset Summary",
                   status = "info",
                   solidHeader = TRUE,
                   width = 12,
                   verbatimTextOutput("data_summary")
                 )
          ),
          column(6,
                 box(
                   title = "Variable Descriptions",
                   status = "info",
                   solidHeader = TRUE,
                   width = 12,
                   tableOutput("variable_descriptions")
                 )
          )
        )
      ),
      
      # Tab 2: Exploratory Analysis
      tabItem(
        tabName = "explore",
        fluidRow(
          column(12,
                 box(
                   title = "Data Distribution",
                   status = "primary",
                   solidHeader = TRUE,
                   width = 12,
                   tabsetPanel(
                     tabPanel("Numeric Variables",
                              fluidRow(
                                column(6, plotlyOutput("age_dist")),
                                column(6, plotlyOutput("bmi_dist"))
                              ),
                              fluidRow(
                                column(6, plotlyOutput("health_score_dist")),
                                column(6, plotOutput("treatment_dist"))
                              )
                     ),
                     tabPanel("Correlation Matrix",
                              plotOutput("correlation_plot"),
                              verbatimTextOutput("correlation_stats")
                     ),
                     tabPanel("Pairwise Relationships",
                              plotlyOutput("pairwise_plot"),
                              selectInput("x_var", "X Variable:",
                                          choices = c("Age", "BMI", "Smoker", "HealthScore"),
                                          selected = "Age"),
                              selectInput("y_var", "Y Variable:",
                                          choices = c("HealthScore", "Age", "BMI"),
                                          selected = "HealthScore"),
                              selectInput("color_var", "Color by:",
                                          choices = c("Treatment", "Smoker", "None"),
                                          selected = "Treatment")
                     )
                   )
                 )
          )
        )
      ),
      
      # Tab 3: Linear Regression
      tabItem(
        tabName = "regression",
        fluidRow(
          column(12,
                 box(
                   title = "Linear Regression Model",
                   status = "success",
                   solidHeader = TRUE,
                   width = 12,
                   tabsetPanel(
                     tabPanel("Model Formula",
                              h4("Regression Model Specification:"),
                              verbatimTextOutput("model_formula"),
                              hr(),
                              h4("Select Variables for Model:"),
                              checkboxGroupInput("model_vars", "Include Variables:",
                                                 choices = c("Age", "BMI", "Smoker", "Treatment"),
                                                 selected = c("Age", "BMI", "Smoker", "Treatment"),
                                                 inline = TRUE),
                              actionButton("update_model", "Update Model", 
                                           class = "btn-primary")
                     ),
                     tabPanel("Model Summary",
                              verbatimTextOutput("model_summary"),
                              h4("Model Diagnostics:"),
                              plotOutput("model_diagnostics")
                     ),
                     tabPanel("Coefficients",
                              DTOutput("coefficients_table"),
                              h4("Interpretation:"),
                              uiOutput("coefficient_interpretation")
                     ),
                     tabPanel("Assumptions Check",
                              h4("Linear Regression Assumptions:"),
                              fluidRow(
                                column(6,
                                       h5("1. Linearity"),
                                       plotOutput("linearity_plot"),
                                       p("Check: Residuals vs Fitted plot should show no pattern")
                                ),
                                column(6,
                                       h5("2. Normality of Residuals"),
                                       plotOutput("normality_plot"),
                                       p("Check: QQ plot should follow diagonal line")
                                )
                              ),
                              fluidRow(
                                column(6,
                                       h5("3. Homoscedasticity"),
                                       plotOutput("homoscedasticity_plot"),
                                       p("Check: Scale-Location plot should show horizontal line")
                                ),
                                column(6,
                                       h5("4. Independence"),
                                       plotOutput("independence_plot"),
                                       p("Check: Residuals vs Order should be random")
                                )
                              )
                     )
                   )
                 )
          )
        )
      ),
      
      # Tab 4: Model Performance
      tabItem(
        tabName = "performance",
        fluidRow(
          column(12,
                 box(
                   title = "Model Performance Metrics",
                   status = "warning",
                   solidHeader = TRUE,
                   width = 12,
                   tabsetPanel(
                     tabPanel("Train vs Test Performance",
                              fluidRow(
                                valueBoxOutput("train_rmse"),
                                valueBoxOutput("test_rmse"),
                                valueBoxOutput("train_r2")
                              ),
                              fluidRow(
                                valueBoxOutput("test_r2"),
                                valueBoxOutput("train_mae"),
                                valueBoxOutput("test_mae")
                              ),
                              hr(),
                              h4("Performance Comparison:"),
                              plotlyOutput("performance_comparison")
                     ),
                     tabPanel("Predictions Visualization",
                              fluidRow(
                                column(6,
                                       h4("Training Set Predictions"),
                                       plotlyOutput("train_predictions")
                                ),
                                column(6,
                                       h4("Test Set Predictions"),
                                       plotlyOutput("test_predictions")
                                )
                              ),
                              fluidRow(
                                column(12,
                                       h4("Residual Analysis"),
                                       plotOutput("residual_analysis")
                                )
                              )
                     ),
                     tabPanel("Error Metrics",
                              h4("Detailed Error Analysis:"),
                              DTOutput("error_metrics_table"),
                              hr(),
                              h4("Error Distribution:"),
                              plotOutput("error_distribution"),
                              h4("Bias-Variance Tradeoff:"),
                              plotOutput("bias_variance_plot")
                     ),
                     tabPanel("Cross-Validation",
                              h4("k-Fold Cross Validation Results:"),
                              numericInput("cv_folds", "Number of Folds:",
                                           min = 3, max = 10, value = 5),
                              actionButton("run_cv", "Run Cross-Validation",
                                           class = "btn-primary"),
                              verbatimTextOutput("cv_results"),
                              plotOutput("cv_plot")
                     )
                   )
                 )
          )
        )
      ),
      
      # Tab 5: Download Reports
      tabItem(
        tabName = "reports",
        fluidRow(
          column(12,
                 box(
                   title = "Export Results",
                   status = "info",
                   solidHeader = TRUE,
                   width = 12,
                   h4("Download Analysis Reports:"),
                   br(),
                   fluidRow(
                     column(4,
                            wellPanel(
                              h5("Dataset Export"),
                              downloadButton("download_data", "Download CSV"),
                              br(), br(),
                              downloadButton("download_codebook", "Download Codebook")
                            )
                     ),
                     column(4,
                            wellPanel(
                              h5("Model Export"),
                              downloadButton("download_model", "Download Model"),
                              br(), br(),
                              downloadButton("download_coef", "Download Coefficients")
                            )
                     ),
                     column(4,
                            wellPanel(
                              h5("Report Export"),
                              downloadButton("download_report", "Download PDF Report"),
                              br(), br(),
                              downloadButton("download_summary", "Download Summary")
                            )
                     )
                   ),
                   hr(),
                   h4("Generate Custom Report:"),
                   textAreaInput("report_notes", "Add Notes to Report:",
                                 rows = 3,
                                 placeholder = "Enter any additional notes or observations..."),
                   actionButton("generate_report", "Generate Report",
                                class = "btn-success btn-lg")
                 )
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  # Reactive values
  healthcare_data <- reactiveVal(NULL)
  train_data <- reactiveVal(NULL)
  test_data <- reactiveVal(NULL)
  lm_model <- reactiveVal(NULL)
  model_predictions <- reactiveVal(NULL)
  cv_results <- reactiveVal(NULL)
  
  # Generate simulated healthcare data
  observeEvent(input$simulate_data, {
    withProgress({
      setProgress(value = 0.2, message = "Generating Dataset", 
                  detail = "Creating patient records...")
      
      # Set seed for reproducibility
      set.seed(123)
      
      # Generate 300 observations
      n <- 300
      
      # Independent variables
      age <- round(runif(n, 18, 80), 0)
      bmi <- round(runif(n, 18, 40), 1)
      smoker <- rbinom(n, 1, 0.3)  # 30% smokers
      treatment <- sample(c("A", "B", "C", "D"), n, replace = TRUE, 
                          prob = c(0.3, 0.3, 0.2, 0.2))
      
      # Create treatment dummy variables (for simulation purposes)
      treatment_dummy <- model.matrix(~treatment - 1)
      
      # Generate health score with realistic relationships
      # Base score: 50
      # Age: -0.3 per year after 30
      # BMI: -1 per unit above 25
      # Smoker: -15 points
      # Treatment effects: A=+10, B=+5, C=0, D=-5
      base_score <- 50
      age_effect <- ifelse(age > 30, (age - 30) * -0.3, 0)
      bmi_effect <- ifelse(bmi > 25, (bmi - 25) * -1, 0)
      smoker_effect <- smoker * -15
      treatment_effect <- case_when(
        treatment == "A" ~ 10,
        treatment == "B" ~ 5,
        treatment == "C" ~ 0,
        treatment == "D" ~ -5,
        TRUE ~ 0
      )
      
      # Random error
      error <- rnorm(n, 0, 5)
      
      # Calculate health score (0-100 scale, clipped)
      health_score <- base_score + age_effect + bmi_effect + smoker_effect + 
        treatment_effect + error
      health_score <- pmin(pmax(round(health_score, 1), 0), 100)
      
      # Create dataframe
      df <- data.frame(
        PatientID = 1:n,
        Age = age,
        BMI = bmi,
        Smoker = factor(smoker, levels = c(0, 1), labels = c("Non-smoker", "Smoker")),
        Treatment = factor(treatment, levels = c("A", "B", "C", "D")),
        HealthScore = health_score,
        AgeEffect = age_effect,
        BMIEffect = bmi_effect,
        SmokerEffect = smoker_effect,
        TreatmentEffect = treatment_effect
      )
      
      healthcare_data(df)
      
      # Split data
      split_data()
      
      setProgress(value = 1.0, message = "Dataset Ready", 
                  detail = paste("Generated", n, "patient records"))
      
    }, min = 0, max = 1, value = 0)
  })
  
  # Split data into train and test
  split_data <- function() {
    df <- healthcare_data()
    if (!is.null(df)) {
      set.seed(456)
      train_index <- createDataPartition(df$HealthScore, 
                                         p = input$train_split/100, 
                                         list = FALSE)
      train_data(df[train_index, ])
      test_data(df[-train_index, ])
    }
  }
  
  # Observe train/test split changes
  observeEvent(input$train_split, {
    split_data()
  })
  
  # Train linear regression model
  observeEvent(input$train_model, {
    req(train_data())
    
    withProgress({
      setProgress(value = 0.3, message = "Training Model", 
                  detail = "Building linear regression...")
      
      # Prepare formula based on selected variables
      vars <- input$model_vars
      if (length(vars) > 0) {
        formula_str <- paste("HealthScore ~", paste(vars, collapse = " + "))
      } else {
        formula_str <- "HealthScore ~ 1"  # Intercept only
      }
      
      # Train model
      model <- lm(as.formula(formula_str), data = train_data())
      lm_model(model)
      
      # Make predictions
      train_pred <- predict(model, newdata = train_data())
      test_pred <- predict(model, newdata = test_data())
      
      predictions <- list(
        train = data.frame(
          Actual = train_data()$HealthScore,
          Predicted = train_pred,
          Residuals = train_data()$HealthScore - train_pred
        ),
        test = data.frame(
          Actual = test_data()$HealthScore,
          Predicted = test_pred,
          Residuals = test_data()$HealthScore - test_pred
        )
      )
      
      model_predictions(predictions)
      
      setProgress(value = 1.0, message = "Model Trained", 
                  detail = "Linear regression completed successfully")
      
    }, min = 0, max = 1, value = 0)
  })
  
  # Update model when variables change
  observeEvent(input$update_model, {
    req(train_data())
    
    # Prepare formula based on selected variables
    vars <- input$model_vars
    if (length(vars) > 0) {
      formula_str <- paste("HealthScore ~", paste(vars, collapse = " + "))
      output$model_formula <- renderText({
        paste("Model Formula:\n", formula_str)
      })
    }
  })
  
  # Data table output
  output$data_table <- renderDT({
    req(healthcare_data())
    datatable(
      healthcare_data() %>% select(PatientID, Age, BMI, Smoker, Treatment, HealthScore),
      options = list(
        pageLength = 10,
        scrollX = TRUE,
        searching = TRUE
      ),
      class = "display compact"
    )
  })
  
  # Data summary
  output$data_summary <- renderPrint({
    req(healthcare_data())
    df <- healthcare_data()
    cat("Dataset Summary:\n")
    cat("================\n")
    cat("Total observations:", nrow(df), "\n")
    cat("Variables:", ncol(df), "\n\n")
    
    cat("Numeric Variables Summary:\n")
    print(summary(df %>% select(Age, BMI, HealthScore)))
    
    cat("\nCategorical Variables:\n")
    cat("Smoker:\n")
    print(table(df$Smoker))
    cat("\nTreatment:\n")
    print(table(df$Treatment))
  })
  
  # Variable descriptions
  output$variable_descriptions <- renderTable({
    data.frame(
      Variable = c("PatientID", "Age", "BMI", "Smoker", "Treatment", "HealthScore"),
      Description = c(
        "Unique patient identifier",
        "Age in years (18-80)",
        "Body Mass Index (18-40)",
        "Smoking status (Non-smoker/Smoker)",
        "Treatment type (A, B, C, D)",
        "Overall health score (0-100, higher is better)"
      ),
      Type = c("Identifier", "Numeric", "Numeric", "Binary", "Categorical", "Numeric")
    )
  })
  
  # Distribution plots
  output$age_dist <- renderPlotly({
    req(healthcare_data())
    p <- ggplot(healthcare_data(), aes(x = Age)) +
      geom_histogram(fill = "#007bff", bins = 20, alpha = 0.7) +
      geom_density(aes(y = ..count..), color = "#0056b3", size = 1) +
      labs(title = "Age Distribution", x = "Age (years)", y = "Count") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$bmi_dist <- renderPlotly({
    req(healthcare_data())
    p <- ggplot(healthcare_data(), aes(x = BMI)) +
      geom_histogram(fill = "#28a745", bins = 20, alpha = 0.7) +
      geom_density(aes(y = ..count..), color = "#1e7e34", size = 1) +
      labs(title = "BMI Distribution", x = "BMI", y = "Count") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$health_score_dist <- renderPlotly({
    req(healthcare_data())
    p <- ggplot(healthcare_data(), aes(x = HealthScore)) +
      geom_histogram(fill = "#ffc107", bins = 20, alpha = 0.7) +
      geom_density(aes(y = ..count..), color = "#e0a800", size = 1) +
      labs(title = "Health Score Distribution", x = "Health Score", y = "Count") +
      theme_minimal()
    ggplotly(p)
  })
  
  output$treatment_dist <- renderPlot({
    req(healthcare_data())
    ggplot(healthcare_data(), aes(x = Treatment, fill = Treatment)) +
      geom_bar() +
      scale_fill_brewer(palette = "Set2") +
      labs(title = "Treatment Type Distribution", x = "Treatment", y = "Count") +
      theme_minimal() +
      theme(legend.position = "none")
  })
  
  # Correlation plot
  output$correlation_plot <- renderPlot({
    req(healthcare_data())
    df_numeric <- healthcare_data() %>%
      mutate(
        SmokerNumeric = as.numeric(Smoker) - 1,
        TreatmentA = as.numeric(Treatment == "A"),
        TreatmentB = as.numeric(Treatment == "B"),
        TreatmentC = as.numeric(Treatment == "C")
      ) %>%
      select(Age, BMI, SmokerNumeric, TreatmentA, TreatmentB, TreatmentC, HealthScore)
    
    cor_matrix <- cor(df_numeric)
    corrplot(cor_matrix, method = "color", type = "upper",
             tl.col = "black", tl.srt = 45,
             addCoef.col = "black",
             number.cex = 0.7,
             col = colorRampPalette(c("blue", "white", "red"))(200))
  })
  
  # Pairwise plot
  output$pairwise_plot <- renderPlotly({
    req(healthcare_data())
    
    # Map variable names
    x_var <- switch(input$x_var,
                    "Age" = "Age",
                    "BMI" = "BMI",
                    "Smoker" = "Smoker",
                    "HealthScore" = "HealthScore")
    
    y_var <- switch(input$y_var,
                    "HealthScore" = "HealthScore",
                    "Age" = "Age",
                    "BMI" = "BMI")
    
    color_var <- ifelse(input$color_var == "None", NULL, input$color_var)
    
    p <- ggplot(healthcare_data(), aes_string(x = x_var, y = y_var, color = color_var)) +
      geom_point(alpha = 0.6, size = 2) +
      geom_smooth(method = "lm", se = FALSE, color = "black") +
      labs(title = paste(y_var, "vs", x_var),
           x = x_var, y = y_var) +
      theme_minimal()
    
    if (!is.null(color_var)) {
      p <- p + scale_color_brewer(palette = "Set1")
    }
    
    ggplotly(p)
  })
  
  # Model summary
  output$model_summary <- renderPrint({
    req(lm_model())
    summary(lm_model())
  })
  
  # Model diagnostics
  output$model_diagnostics <- renderPlot({
    req(lm_model())
    par(mfrow = c(2, 2))
    plot(lm_model())
  })
  
  # Coefficients table
  output$coefficients_table <- renderDT({
    req(lm_model())
    coef_df <- tidy(lm_model())
    datatable(
      coef_df,
      options = list(
        pageLength = 10,
        searching = FALSE
      ),
      colnames = c("Term", "Estimate", "Std.Error", "t-value", "p-value"),
      class = "display compact"
    ) %>%
      formatRound(columns = c("estimate", "std.error", "statistic"), digits = 3) %>%
      formatSignif(columns = "p.value", digits = 3)
  })
  
  # Coefficient interpretation
  output$coefficient_interpretation <- renderUI({
    req(lm_model())
    coefs <- tidy(lm_model())
    
    interpretations <- lapply(1:nrow(coefs), function(i) {
      term <- coefs$term[i]
      estimate <- coefs$estimate[i]
      p_value <- coefs$p.value[i]
      
      significance <- ifelse(p_value < 0.001, "highly significant",
                             ifelse(p_value < 0.01, "very significant",
                                    ifelse(p_value < 0.05, "significant",
                                           ifelse(p_value < 0.1, "marginally significant", "not significant"))))
      
      effect_direction <- ifelse(estimate > 0, "increases", "decreases")
      
      tagList(
        p(strong(term), ":"),
        p("• Coefficient: ", round(estimate, 3)),
        p("• Effect: A one-unit increase in ", term, " ", effect_direction, 
          " Health Score by ", abs(round(estimate, 3)), " points"),
        p("• Significance: ", significance, " (p = ", round(p_value, 4), ")"),
        hr()
      )
    })
    
    tagList(
      h5("Interpretation of Coefficients:"),
      interpretations
    )
  })
  
  # Assumptions check plots
  output$linearity_plot <- renderPlot({
    req(lm_model())
    plot(lm_model(), which = 1)
  })
  
  output$normality_plot <- renderPlot({
    req(lm_model())
    plot(lm_model(), which = 2)
  })
  
  output$homoscedasticity_plot <- renderPlot({
    req(lm_model())
    plot(lm_model(), which = 3)
  })
  
  output$independence_plot <- renderPlot({
    req(lm_model())
    residuals <- resid(lm_model())
    plot(residuals, type = "b", 
         xlab = "Observation Order", ylab = "Residuals",
         main = "Residuals vs Order")
    abline(h = 0, col = "red")
  })
  
  # Performance metrics value boxes
  output$train_rmse <- renderValueBox({
    req(model_predictions())
    rmse <- sqrt(mean(model_predictions()$train$Residuals^2))
    valueBox(
      value = round(rmse, 2),
      subtitle = "Train RMSE",
      icon = icon("calculator"),
      color = "blue"
    )
  })
  
  output$test_rmse <- renderValueBox({
    req(model_predictions())
    rmse <- sqrt(mean(model_predictions()$test$Residuals^2))
    valueBox(
      value = round(rmse, 2),
      subtitle = "Test RMSE",
      icon = icon("calculator"),
      color = ifelse(rmse < 10, "green", 
                     ifelse(rmse < 15, "yellow", "red"))
    )
  })
  
  output$train_r2 <- renderValueBox({
    req(lm_model())
    r2 <- summary(lm_model())$r.squared
    valueBox(
      value = round(r2, 3),
      subtitle = "Train R-squared",
      icon = icon("chart-line"),
      color = "blue"
    )
  })
  
  output$test_r2 <- renderValueBox({
    req(model_predictions())
    ss_res <- sum(model_predictions()$test$Residuals^2)
    ss_tot <- sum((model_predictions()$test$Actual - mean(model_predictions()$test$Actual))^2)
    r2 <- 1 - (ss_res/ss_tot)
    valueBox(
      value = round(r2, 3),
      subtitle = "Test R-squared",
      icon = icon("chart-line"),
      color = ifelse(r2 > 0.7, "green", 
                     ifelse(r2 > 0.5, "yellow", "red"))
    )
  })
  
  output$train_mae <- renderValueBox({
    req(model_predictions())
    mae <- mean(abs(model_predictions()$train$Residuals))
    valueBox(
      value = round(mae, 2),
      subtitle = "Train MAE",
      icon = icon("ruler"),
      color = "blue"
    )
  })
  
  output$test_mae <- renderValueBox({
    req(model_predictions())
    mae <- mean(abs(model_predictions()$test$Residuals))
    valueBox(
      value = round(mae, 2),
      subtitle = "Test MAE",
      icon = icon("ruler"),
      color = ifelse(mae < 8, "green", 
                     ifelse(mae < 12, "yellow", "red"))
    )
  })
  
  # Performance comparison plot
  output$performance_comparison <- renderPlotly({
    req(model_predictions())
    
    metrics <- data.frame(
      Dataset = rep(c("Train", "Test"), each = 3),
      Metric = rep(c("RMSE", "MAE", "R-squared"), 2),
      Value = c(
        sqrt(mean(model_predictions()$train$Residuals^2)),
        mean(abs(model_predictions()$train$Residuals)),
        summary(lm_model())$r.squared,
        sqrt(mean(model_predictions()$test$Residuals^2)),
        mean(abs(model_predictions()$test$Residuals)),
        1 - (sum(model_predictions()$test$Residuals^2) / 
               sum((model_predictions()$test$Actual - mean(model_predictions()$test$Actual))^2))
      )
    )
    
    p <- ggplot(metrics, aes(x = Metric, y = Value, fill = Dataset)) +
      geom_bar(stat = "identity", position = "dodge") +
      scale_fill_brewer(palette = "Set1") +
      labs(title = "Model Performance: Train vs Test",
           x = "Metric", y = "Value") +
      theme_minimal() +
      theme(legend.position = "top")
    
    ggplotly(p)
  })
  
  # Train predictions plot
  output$train_predictions <- renderPlotly({
    req(model_predictions())
    
    plot_data <- model_predictions()$train
    correlation <- cor(plot_data$Actual, plot_data$Predicted)
    
    p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.6, color = "#007bff") +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      geom_smooth(method = "lm", se = FALSE, color = "green") +
      labs(title = paste("Training Set: Actual vs Predicted (r =", round(correlation, 3), ")"),
           x = "Actual Health Score", y = "Predicted Health Score") +
      theme_minimal() +
      coord_equal()
    
    ggplotly(p)
  })
  
  # Test predictions plot
  output$test_predictions <- renderPlotly({
    req(model_predictions())
    
    plot_data <- model_predictions()$test
    correlation <- cor(plot_data$Actual, plot_data$Predicted)
    
    p <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.6, color = "#28a745") +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      geom_smooth(method = "lm", se = FALSE, color = "green") +
      labs(title = paste("Test Set: Actual vs Predicted (r =", round(correlation, 3), ")"),
           x = "Actual Health Score", y = "Predicted Health Score") +
      theme_minimal() +
      coord_equal()
    
    ggplotly(p)
  })
  
  # Residual analysis
  output$residual_analysis <- renderPlot({
    req(model_predictions())
    
    plot_data <- rbind(
      cbind(model_predictions()$train, Dataset = "Train"),
      cbind(model_predictions()$test, Dataset = "Test")
    )
    
    ggplot(plot_data, aes(x = Predicted, y = Residuals, color = Dataset)) +
      geom_point(alpha = 0.6) +
      geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
      geom_smooth(se = FALSE, method = "loess") +
      scale_color_manual(values = c("Train" = "#007bff", "Test" = "#28a745")) +
      labs(title = "Residual Analysis by Dataset",
           x = "Predicted Health Score", y = "Residuals") +
      theme_minimal() +
      facet_wrap(~Dataset, scales = "free") +
      theme(legend.position = "none")
  })
  
  # Error metrics table
  output$error_metrics_table <- renderDT({
    req(model_predictions())
    
    calculate_metrics <- function(residuals, actual, predicted) {
      data.frame(
        RMSE = sqrt(mean(residuals^2)),
        MAE = mean(abs(residuals)),
        MAPE = mean(abs(residuals/actual)) * 100,
        R_squared = 1 - (sum(residuals^2) / sum((actual - mean(actual))^2)),
        Adjusted_R2 = 1 - ((1 - (1 - (sum(residuals^2) / sum((actual - mean(actual))^2))) * 
                              (length(actual) - 1)) / (length(actual) - 5 - 1))
      )
    }
    
    train_metrics <- calculate_metrics(
      model_predictions()$train$Residuals,
      model_predictions()$train$Actual,
      model_predictions()$train$Predicted
    )
    
    test_metrics <- calculate_metrics(
      model_predictions()$test$Residuals,
      model_predictions()$test$Actual,
      model_predictions()$test$Predicted
    )
    
    metrics_df <- data.frame(
      Metric = c("RMSE", "MAE", "MAPE (%)", "R-squared", "Adjusted R-squared"),
      Train = round(unlist(train_metrics), 3),
      Test = round(unlist(test_metrics), 3),
      Difference = round(unlist(train_metrics) - unlist(test_metrics), 3)
    )
    
    datatable(
      metrics_df,
      options = list(
        pageLength = 10,
        searching = FALSE
      ),
      class = "display compact"
    ) %>%
      formatStyle(
        'Difference',
        backgroundColor = styleInterval(
          c(-0.5, 0.5),
          c('#d4edda', '#fff3cd', '#f8d7da')
        )
      )
  })
  
  # Error distribution
  output$error_distribution <- renderPlot({
    req(model_predictions())
    
    errors <- data.frame(
      Residuals = c(model_predictions()$train$Residuals, 
                    model_predictions()$test$Residuals),
      Dataset = rep(c("Train", "Test"), 
                    c(nrow(model_predictions()$train), 
                      nrow(model_predictions()$test)))
    )
    
    ggplot(errors, aes(x = Residuals, fill = Dataset)) +
      geom_histogram(alpha = 0.6, position = "identity", bins = 30) +
      geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
      scale_fill_manual(values = c("Train" = "#007bff", "Test" = "#28a745")) +
      labs(title = "Distribution of Residuals",
           x = "Residuals", y = "Frequency") +
      theme_minimal() +
      facet_wrap(~Dataset, scales = "free_y")
  })
  
  # Cross-validation
  observeEvent(input$run_cv, {
    req(train_data())
    
    withProgress({
      setProgress(value = 0.3, message = "Running Cross-Validation", 
                  detail = "Performing k-fold CV...")
      
      # Prepare data
      df <- train_data()
      
      # Set up k-fold cross-validation
      train_control <- trainControl(
        method = "cv",
        number = input$cv_folds,
        savePredictions = TRUE
      )
      
      # Train model with CV
      cv_model <- train(
        HealthScore ~ Age + BMI + Smoker + Treatment,
        data = df,
        method = "lm",
        trControl = train_control
      )
      
      cv_results(cv_model)
      
      setProgress(value = 1.0, message = "CV Complete", 
                  detail = paste("Completed", input$cv_folds, "-fold cross-validation"))
      
    }, min = 0, max = 1, value = 0)
  })
  
  # CV results
  output$cv_results <- renderPrint({
    req(cv_results())
    cat("Cross-Validation Results:\n")
    cat("=========================\n\n")
    cat("Method:", cv_results()$method, "\n")
    cat("Number of folds:", cv_results()$control$number, "\n")
    cat("\nPerformance Metrics across folds:\n")
    print(cv_results()$results)
    cat("\nFinal Model Performance (Averaged across folds):\n")
    print(cv_results()$finalModel)
  })
  
  # Download handlers
  output$download_data <- downloadHandler(
    filename = function() {
      paste("healthcare_data_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      write.csv(healthcare_data(), file, row.names = FALSE)
    }
  )
  
  output$download_model <- downloadHandler(
    filename = function() {
      paste("linear_regression_model_", Sys.Date(), ".rds", sep = "")
    },
    content = function(file) {
      req(lm_model())
      saveRDS(lm_model(), file)
    }
  )
  
  # Initialize with sample data
  observe({
    if (is.null(healthcare_data())) {
      input$simulate_data
    }
  })
  
}

shinyApp(ui, server)