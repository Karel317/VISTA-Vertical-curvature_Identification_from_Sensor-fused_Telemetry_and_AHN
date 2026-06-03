import pyzed.sl as sl
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import Decoder
import struct

# --- INSTELLINGEN ---
SVO_PATH = r"D:\29 april\14_20_00\front_01_05_2026-14_20_00.svo2"
MCAP_PATH = r"D:\29 april\14_20_00\rosbag\rosbag_0.mcap"
# Nieuwe bestandsnaam om verwarring te voorkomen
OUTPUT_BIN = r"C:\Users\Lars Wissink\OneDrive\Documenten\lars wissink\WB TU Delft\jaar 3\bep\.mcap to Bin downloads\full_dataset_cleaned.bin"
TARGET_FRAME = 407
LIDAR_TOPICS = ["/rslidar/M1P_deskewed"]
LIDAR_Y_OFFSET_IN_ZED_WORLD = 0

def get_removal_mask(points, box_3d, y_offset, buf=0.1):
    """
    Geeft een boolean array terug: True voor punten die BINNEN de box vallen.
    """
    x_min, y_min, z_min = np.min(box_3d, axis=0)
    x_max, y_max, z_max = np.max(box_3d, axis=0)
    
    # Assen swap conform jouw werkende mapping
    p_zed_x = -points[:, 1]           
    p_zed_z = points[:, 0]            
    p_zed_y = -points[:, 2] + y_offset 

    inside = (p_zed_x >= x_min - buf) & (p_zed_x <= x_max + buf) & \
             (p_zed_y >= y_min - buf) & (p_zed_y <= y_max + buf) & \
             (p_zed_z >= z_min - buf) & (p_zed_z <= z_max + buf)
    return inside

def main():
    # 1. ZED Detectie (Boxen ophalen)
    zed = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(SVO_PATH)
    init_p = sl.InitParameters(input_t=input_type, depth_mode=sl.DEPTH_MODE.NEURAL, coordinate_units=sl.UNIT.METER)
    
    if zed.open(init_p) != sl.ERROR_CODE.SUCCESS:
        print("SVO openen mislukt.")
        return

    zed.set_svo_position(TARGET_FRAME)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)
    
    objects = sl.Objects()
    object_boxes = []
    ts_ns = 0

    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        ts_ns = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        zed.retrieve_objects(objects, sl.ObjectDetectionRuntimeParameters())
        for obj in objects.object_list:
            if obj.label in [sl.OBJECT_CLASS.PERSON, sl.OBJECT_CLASS.VEHICLE]:
                object_boxes.append(obj.bounding_box)
                print(f"Camera herkent {obj.label} (ID: {obj.id}) voor verwijdering.")
    zed.close()

    if not object_boxes:
        print("Geen objecten gevonden om te verwijderen. Dataset blijft ongewijzigd.")

    # 2. Lidar Extractie (Volledige dataset ophalen)
    all_lidar_points = []
    with open(MCAP_PATH, "rb") as f:
        reader = make_reader(f)
        decoder = Decoder()
        for topic in LIDAR_TOPICS:
            best_msg = None
            min_diff = float('inf')
            for schema, channel, msg in reader.iter_messages(topics=[topic], start_time=ts_ns-100_000_000, end_time=ts_ns+100_000_000):
                diff = abs(msg.log_time - ts_ns)
                if diff < min_diff:
                    min_diff = diff
                    best_msg, best_schema = msg, schema
            
            if best_msg:
                ros_msg = decoder.decode(best_schema, best_msg)
                data = np.frombuffer(ros_msg.data, dtype=np.uint8)
                num_p = ros_msg.width * ros_msg.height
                topic_pts = np.zeros((num_p, 5), dtype=np.float32)
                for i in range(num_p):
                    offset = i * ros_msg.point_step
                    p = struct.unpack_from('ffff', data, offset)
                    r = struct.unpack_from('H', data, offset + 16)[0]
                    # ... (bovenkant van je main blijft gelijk) ...

            if best_msg:
                ros_msg = decoder.decode(best_schema, best_msg)
                data = np.frombuffer(ros_msg.data, dtype=np.uint8)
                num_p = ros_msg.width * ros_msg.height
                topic_pts = np.zeros((num_p, 5), dtype=np.float32)
                
                for i in range(num_p):
                    offset = i * ros_msg.point_step
                    # Hier worden de rauwe x, y, z uit de data gehaald
                    p = struct.unpack_from('ffff', data, offset)
                    r = struct.unpack_from('H', data, offset + 16)[0]
                    
                    # --- DIT STUKJE VOEG JE TOE / PAS JE AAN ---
                    lx, ly, lz = p[0], p[1], p[2]

                    # Controleer of dit een Helios sensor is die 'spookt'
                    if "helios" in topic:
                        # Als de Helios achteruit kijkt, draaien we X en Y om
                        # Probeer eerst dit om de 'dubbele' mensen weg te krijgen
                        lx = -lx
                        ly = -ly 

                    topic_pts[i] = [lx, ly, lz, p[3], float(r)]
                    # ------------------------------------------

                
                all_lidar_points.append(topic_pts)

    if not all_lidar_points:
        print("Geen Lidar data gevonden.")
        return

    # Combineer alle Lidar data tot de volledige dataset
    full_dataset = np.vstack(all_lidar_points)
    print(f"Oorspronkelijke dataset bevat {len(full_dataset)} punten.")

   
    # 3. Filteren (De "Grote Gum" methode)
    if object_boxes:
        print(f"Bezig met wegvlakken van {len(object_boxes)} objecten uit de hoofdcloud...")
        
        # We maken een kopie van de volledige dataset
        cleaned_dataset = full_dataset.copy()
        
        for i, box in enumerate(object_boxes):
            # Vergroot de buffer naar 0.30 meter om zeker te weten dat we alles raken
            removal_mask = get_removal_mask(cleaned_dataset, box, LIDAR_Y_OFFSET_IN_ZED_WORLD, buf=0.50)
            
            # Tel hoeveel punten we gaan verwijderen voor dit object
            points_to_remove = np.sum(removal_mask)
            print(f" -> Object {i}: {points_to_remove} punten gemarkeerd voor verwijdering.")
            
            # Behoud alleen wat NIET in de box zit
            cleaned_dataset = cleaned_dataset[~removal_mask]
    else:
        cleaned_dataset = full_dataset
        print("Geen objecten gevonden om te verwijderen.")

    # 4. Opslaan
    cleaned_dataset.tofile(OUTPUT_BIN)
    
    removed_count = len(full_dataset) - len(cleaned_dataset)
    print(f"\nKlaar!")
    print(f"Verwijderd: {removed_count} punten (personen/voertuigen).")
    print(f"Overgebleven: {len(cleaned_dataset)} punten.")
    print(f"Bestand opgeslagen als: {OUTPUT_BIN}")

if __name__ == "__main__":
    main()