# modules/vision/detect/include/SushiAI/vision/detect/iou_gradient.hpp


## BoxGradient, complete_iou_gradient

`complete_iou_gradient(box, target, form, gradient)` writes the gradient of the complete IoU of
two `x_centre y_centre width height` boxes at `box`. `gradient` receives four components in the
order of `box`, and the measure rises along them. All four are 0 when both boxes are one point.
Each span must hold four values.

`BoxGradient` names the form. `Analytic` is the derivative of the measure. `DarknetCompatible`
is the gradient section 11 of `docs/agent/2026_10_09_DARKNET_YOLO/REPORT.md` states.
`K_ASPECT_TRADE_OFF_EPS` is 1e-6, the term that keeps the trade-off of the aspect penalty
finite.

`test_detect_iou_gradient.cpp` compares the analytic form with a central difference of step
1e-6, within 1e-6 relative plus 1e-7 absolute, on fixed pairs and on 100 seeded pairs.
