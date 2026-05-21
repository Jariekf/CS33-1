# Business Insight Report

## Executive Answer
The workflow completed the analysis for: Predict saleprice.

## Key Evidence
- Task type: classification
- Target column: id
- Dataset description provided: True
- Plan version: 3
- EDA target summary: 1168 classes; majority ratio 0.00

## Model Performance
- Best model: logistic_regression
- Primary metric: f1_weighted
- Test metrics: {'accuracy': 0.0, 'f1_weighted': 0.0}

## Business Recommendations
- Review the top drivers and validate whether they are actionable in the business process.
- Use the model output as decision support, not as an automatic decision rule.
- Re-run the workflow with richer metadata if the target definition or dataset meaning is unclear.

## Top Features
- mosold: 0.4791
- bsmtfullbath: 0.2916
- masvnrtype_was_missing: 0.2909
- wooddecksf: 0.2722
- bsmtunfsf: 0.2682
- lotshape_IR1: 0.2662
- lotshape_Reg: 0.2634
- totrmsabvgrd: 0.2462
- fireplacequ_was_missing: 0.2414
- garagefinish_RFn: 0.2275

## EDA Findings
- poolarea: abs_corr_with_target_code=0.056
- lowqualfinsf: abs_corr_with_target_code=0.0508
- 3ssnporch: abs_corr_with_target_code=0.0494

## Limitations and Next Steps
- LLM report generation was unavailable, so this fallback report was used. Reason: Error code: 400 - {'error': {'message': "This model's maximum context length is 128000 tokens. However, your messages resulted in 203137 tokens. Please reduce the length of the messages.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}
- Confirm metric suitability and evaluate the model on more representative data before deployment.