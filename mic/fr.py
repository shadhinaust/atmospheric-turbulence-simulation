IMAGES_DOWNLOAD_DIRECTORY = "./data/lfw-py"
IMAGES_DIRECTORY = "images/faces"

people_number = []
for person in people:
    folder_path = IMAGES_DOWNLOAD_DIRECTORY + '/lfw-deepfunneled/' + person
    num_images = len(os.listdir(folder_path))
    people_number.append((person, num_images))
 
people_number = sorted(people_number, key=lambda x: x[1], reverse=True)
people_with_one_photo = [(person) for person, num_images in people_number if num_images==1]
print("Individuals with one photo: {}".format(len(people_with_one_photo)))