import pandas as pd
from pathlib import Path
from typing import List


def predictions2submission(results: List, submission_path: Path):
    predictions = pd.DataFrame(columns=["filename", "bbox"])

    for r in results:
        filename = Path(r.path).name
        bbox_strings = []
        for box in r.boxes:
            # 0 x_center y_center width height
            bbox = box.xywhn[0].tolist()
            bbox_strings.append("0" + " " + " ".join(map(str, bbox)))
        
        if len(bbox_strings) > 0:
            full_string = " ".join(bbox_strings)
        else:
            full_string = "-1"
        predictions.loc[len(predictions)] = [filename, full_string]

    predictions.reset_index().to_csv(submission_path, index=False)
