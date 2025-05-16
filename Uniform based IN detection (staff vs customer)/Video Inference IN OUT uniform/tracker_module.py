"""
Object tracking utilities with motion tracking.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_centroid(box):
    """
    Compute the centroid of a bounding box.
    
    Args:
        box: Bounding box in format [x1, y1, x2, y2]
        
    Returns:
        np.array: Centroid coordinates [cx, cy]
    """
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def unit_vector(v, eps=1e-6):
    """
    Compute the unit vector of a given vector.
    
    Args:
        v: Input vector
        eps: Small value to avoid division by zero
        
    Returns:
        np.array or None: Unit vector or None if norm is too small
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > eps else None

def compute_iou(box, boxes):
    """
    Compute IoU between one box and multiple boxes.
    
    Args:
        box: Single bounding box
        boxes: Array of bounding boxes
        
    Returns:
        tuple: (max_iou, best_match_index)
    """
    if box is None or len(box) != 4 or len(boxes) == 0:
        return None, None

    box = np.array(box)
    boxes = np.array(boxes)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area

    ious = inter_area / np.maximum(union_area, 1e-6)

    max_iou = np.max(ious)
    best_match_idx = np.argmax(ious)

    return max_iou, best_match_idx

class ObjectTracker:
    """Object tracker with motion tracking and direction estimation."""
    
    def __init__(self, iou_threshold=0.2, max_dist=100, expire_after=120, BASE_IOU_W=0.9, BASE_DIR_W=0.1):
        """
        Initialize the tracker.
        
        Args:
            iou_threshold: IoU threshold for matching
            max_dist: Maximum distance for centroid matching (in pixels)
            expire_after: Number of frames after which to expire a track
        """
        self.iou_threshold = iou_threshold
        self.max_dist = max_dist
        self.expire_after = expire_after

        self.BASE_IOU_W = BASE_IOU_W
        self.BASE_DIR_W = BASE_DIR_W
        
        # Tracking state
        self.tracked_objects = {}    # track_id → box
        self.last_seen = {}          # track_id → last seen frame
        self.centroids = {}          # track_id → last centroid (np.array)
        self.directions = {}         # track_id → last unit-vector (np.array) or None
        self.next_id = 0
    
    def update(self, boxes, frame_count):
        """
        Update tracks with new detections.
        
        Args:
            boxes: Array of bounding boxes in format [x1, y1, x2, y2]
            frame_count: Current frame count
            
        Returns:
            dict: Current frame's active tracks {track_id: box}
        """
        # 1. Purge stale tracks
        stale = [tid for tid, last in self.last_seen.items()
                if frame_count - last > self.expire_after]
        for tid in stale:
            self.tracked_objects.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.centroids.pop(tid, None)
            self.directions.pop(tid, None)

        prev_ids = list(self.tracked_objects.keys())
        prev_boxes = [self.tracked_objects[tid] for tid in prev_ids]
        curr_boxes = list(boxes)
        curr_cents = [compute_centroid(b) for b in curr_boxes]

        P, C = len(prev_boxes), len(curr_boxes)
        assigned_prev = set()
        assigned_curr = set()
        new_assignments = {}

        if P > 0 and C > 0:
            # Precompute prev centroids & directions
            prev_cents = np.array([self.centroids[tid] for tid in prev_ids])
            prev_dirs = [self.directions.get(tid) for tid in prev_ids]

            iou_mat = np.zeros((P, C), dtype=np.float32)
            dir_mat = np.zeros((P, C), dtype=np.float32)
            cost_mat = np.ones((P, C), dtype=np.float32)

            for i, p in enumerate(prev_boxes):
                p = np.array(p)
                cb = np.array(curr_boxes)
                
                # IoU computation
                x1 = np.maximum(p[0], cb[:, 0])
                y1 = np.maximum(p[1], cb[:, 1])
                x2 = np.minimum(p[2], cb[:, 2])
                y2 = np.minimum(p[3], cb[:, 3])
                
                inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                area_p = (p[2] - p[0]) * (p[3] - p[1])
                area_c = (cb[:, 2] - cb[:, 0]) * (cb[:, 3] - cb[:, 1])
                union = area_p + area_c - inter
                
                iou_mat[i] = inter / np.maximum(union, 1e-6)

                # Distance gating + direction
                for j, cc in enumerate(curr_cents):
                    # Distance gating
                    dist = np.linalg.norm(cc - prev_cents[i])
                    if dist > self.max_dist:
                        continue  # cost stays at 1 → effectively forbidden

                    # Direction score
                    prev_dir = prev_dirs[i]
                    delta = cc - prev_cents[i]
                    cand_dir = unit_vector(delta)
                    
                    if prev_dir is not None and cand_dir is not None:
                        dir_mat[i, j] = max(0.0, np.dot(prev_dir, cand_dir))
                    # otherwise dir_mat[i,j] stays 0

                    # Combined cost (equal weights)
                    score = self.BASE_IOU_W * iou_mat[i, j] + self.BASE_DIR_W * dir_mat[i, j]
                    cost_mat[i, j] = 1.0 - score

            # Hungarian on fused cost
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            # Accept only good matches
            for r, c in zip(row_ind, col_ind):
                if iou_mat[r, c] >= self.iou_threshold:
                    pid = prev_ids[r]
                    assigned_prev.add(pid)
                    assigned_curr.add(c)
                    new_assignments[pid] = curr_boxes[c]

        # Update matched tracks
        for pid, box in new_assignments.items():
            self.tracked_objects[pid] = box
            self.last_seen[pid] = frame_count

            # Update centroid & direction
            new_c = compute_centroid(box)
            old_c = self.centroids[pid]
            self.centroids[pid] = new_c
            
            delta = new_c - old_c
            u = unit_vector(delta)
            if u is not None:
                self.directions[pid] = u

        # Spawn new IDs
        for idx, box in enumerate(curr_boxes):
            if idx in assigned_curr:
                continue
                
            pid = self.next_id
            self.tracked_objects[pid] = box
            self.last_seen[pid] = frame_count
            self.centroids[pid] = compute_centroid(box)
            self.directions[pid] = None  # No direction yet
            self.next_id += 1

        # Return current-frame tracks
        return {
            tid: self.tracked_objects[tid]
            for tid, last in self.last_seen.items()
            if last == frame_count
        }