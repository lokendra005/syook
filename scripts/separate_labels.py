import os
import shutil

def separate_labels(input_dir, output_dir_person, output_dir_ppe):
    """Separate person and PPE labels into different directories."""
    if not os.path.exists(output_dir_person):
        os.makedirs(output_dir_person)
    
    if not os.path.exists(output_dir_ppe):
        os.makedirs(output_dir_ppe)
    
    for label_file in os.listdir(input_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(input_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()

            person_lines = []
            ppe_lines = []
            for line in lines:
                class_id, *rest = line.strip().split()
                class_id = int(class_id)
                
                if class_id == 0:  # Assuming 'person' has class index 0
                    person_lines.append(line)
                else:
                    # Decrement the class ID for PPE classes
                    new_class_id = class_id - 1
                    ppe_lines.append(f"{new_class_id} {' '.join(rest)}\n")
            
            # Save the person labels
            if person_lines:
                person_label_path = os.path.join(output_dir_person, label_file)
                with open(person_label_path, 'w') as f:
                    f.writelines(person_lines)
            
            # Save the PPE labels
            if ppe_lines:
                ppe_label_path = os.path.join(output_dir_ppe, label_file)
                with open(ppe_label_path, 'w') as f:
                    f.writelines(ppe_lines)

def main():
    input_dir = "D:\syookAssignment\outputs\Yolo_labels"
    output_dir_person = "D:\syookAssignment\outputs\separated_labels/person"
    output_dir_ppe = "D:\syookAssignment\outputs\separated_labels/ppe"
    
    separate_labels(input_dir, output_dir_person, output_dir_ppe)

if __name__ == "__main__":
    main()
















