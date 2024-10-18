import socket
import json
import datetime
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, cast, List
from typing_extensions import Self
from viam.components.sensor import Sensor
from viam.components.camera import Camera, ViamImage
from viam.proto.service.vision import Classification, Detection, GetPropertiesResponse
from viam.services.vision import Vision, CaptureAllResult
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict, from_dm_from_extra, ValueTypes
from viam.errors import NoCaptureToStoreError

LOGGER = getLogger(__name__)

class MySensor(Sensor):
    MODEL: ClassVar[Model] = Model(ModelFamily("bill", "detections"), "timer")
    
    def __init__(self, name: str):
        super().__init__(name)

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        sensor = cls(config.name)
        sensor.reconfigure(config, dependencies)
        return sensor

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> Sequence[str]:
        return []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        LOGGER.info("Reconfiguring " + self.name)
        self.config = config
        self.DEPS = dependencies
        config_dict = struct_to_dict(config.attributes)
        self.base_vision_name = config_dict["base_vision_name"]
        self.base_camera_name = config_dict["base_camera_name"]
        self.valid_labels = config_dict["valid_labels"]
        self.label_confidence = config_dict["label_confidence"]
        self.hold_time_threshold = config_dict["hold_time_threshold"]
        pass

    async def get_model_detection(
        self,
        vision_name: str,
        camera_name: str,
    ) -> Detection:
        actual_model = self.DEPS[Vision.get_resource_name(vision_name)]
        vision = cast(Vision, actual_model)
        actual_cam = self.DEPS[Camera.get_resource_name(camera_name)]
        cam = cast(Camera, actual_cam)
        detections = await vision.get_detections_from_camera(camera_name)

        return detections
    
    async def get_readings(self, extra: Optional[Dict[str, Any]] = None, **kwargs) -> Mapping[str, Any]:
        # Counters for the number of detections before and after the hold time
        after_or_equal_hold_time_count = 0
        before_hold_time_count = 0

        # Get detections using the get_model_detection method
        detections = await self.get_model_detection(self.base_vision_name, self.base_camera_name)

        for detection in detections:
            # Extract the timestamp from the class_name
            class_name_parts = detection.class_name.split('_')
            if len(class_name_parts) < 4:
                LOGGER.warning(f"Unexpected class_name format: {detection.class_name}")
                continue

            if class_name_parts[0] not in self.valid_labels:
                break

            # Extract date and time from the class_name
            date_str = class_name_parts[2]
            time_str = class_name_parts[3]

            # Combine date and time into a single string and parse it
            datetime_str = f"{date_str} {time_str}"
            detection_time = datetime.datetime.strptime(datetime_str, "%Y%m%d %H%M%S")

            # Check if the detection time is greater than the hold_time_threshold
            current_time = datetime.datetime.now()
            time_difference = (current_time - detection_time).total_seconds()


            # Count the detection based on the threshold
            if time_difference > self.hold_time_threshold:
                after_or_equal_hold_time_count += 1
            else:
                before_hold_time_count += 1


        # Return the counts in the specified format
        return {
            "pizzasExceedingHoldTime": after_or_equal_hold_time_count,
            "pizzasNotExceedingHoldTime": before_hold_time_count,
            "totalPizzas": after_or_equal_hold_time_count + before_hold_time_count
        }
