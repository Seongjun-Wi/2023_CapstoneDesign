# 2023_CapstoneDesign

이 레포지토리는 소프트웨어융합학과 17학번 위성준의 2023년도 캡스톤디자인 입니다.
해당 캡스톤디자인에서는 저비용 모션캡처를 구현합니다.

사용자는 IR LED Active Marker가 장착된 슈트를 착용하고 IR Pass Filter를 장착한 카메라를 사용하여 움직임을 캡처합니다.
움직임을 캡처할 때는
  1. 마커에서 나온 빛이 적외선 카메라의 광각 중심 방향으로 적외선이 조사 되도록 합니다.
  2. (가시광선이 차단 된 상태에서 상대적으로) 많이 조사된 마커의 빛은 렌즈에 도달하여 카메라가 촬영하게 됩니다.
  3. 촬영한 결과물을 보면 다른 것보다 마커의 빛이 강하게 출력됩니다.
해당 과정을 통해 카메라로 움직임을 캡처할 수 있습니다.

캘리브레이션은 기하학적으로 실제 환경을 모델링하여 미지수 값(내부, 외부 파라미터)를 찾아내는 과정입니다.
단순히 개별 카메라를 통해 chess board와 같은 피사체를 촬영하여 각각의 내부 파라미터를 구할 수는 있습니다. 그러나 외부 파라미터는 모든 카메라가 종속된 하나의 공간에 대해서 구해야 합니다.
따라서 방향이 구분되지 않는 chess board보다는, 방향이 구분되면서 planar한 물체를 calibration의 피사체로 활용해야 합니다.
그렇기에 저는 ELD를 사용하여 Planar한 판을 제작하여 사용하였습니다.
