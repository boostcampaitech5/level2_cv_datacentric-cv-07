import json
import numpy as np

## Scale coordinates (0~1)
SCALE_FACTOR = 5000

def seed_everything(seed):
    np.random.seed(seed)

def main():
        
    seed_everything(1234)
    
    # COCO dataset path
    input_file = input('COCO format json file path : ')
    # Output dataset(UFO format) path
    output_file = "UFO_format_output.json"

    # Resolution scale factor(Default : 1)
    SCALE_FACTOR = 1
    SCALE_FACTOR = int(input('scale factor(default:1) : '))

    # Load json
    with open(input_file, "r", encoding='utf-8') as f:
        data = json.load(f)


    output_data = {
        "images": {}
    }

    # Image name
    for image_info in data["images"]:
        image_id = image_info["id"]
        file_name = image_info["file_name"]

        # COCO annotations -> UFO annotations
        annotations = {}
        for index, annotation in enumerate(data["annotations"]):
            if annotation["image_id"] == image_id:
                annotation_id = annotation["id"]
                bbox = annotation["bbox"]
                points = [
                    [(bbox[0])*SCALE_FACTOR, bbox[1]*SCALE_FACTOR],
                    [(bbox[0] + bbox[2])*SCALE_FACTOR, bbox[1]*SCALE_FACTOR],
                    [(bbox[0] + bbox[2])*SCALE_FACTOR, (bbox[1] + bbox[3])*SCALE_FACTOR],
                    [bbox[0]*SCALE_FACTOR, (bbox[1] + bbox[3])*SCALE_FACTOR]
                ]
                annotations[str(index+1).zfill(4)] = {
                "points": points,
                "tags" : "Auto",
                "illegibility": False
                
                }

        # Assign annotations on each images
        output_data["images"][file_name] = {
            "words": annotations
        }

    # Create UFO format file in working directory
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f,indent=4,ensure_ascii=False)

    print("json file created!")
    print("")

if __name__ == '__main__':
    main()