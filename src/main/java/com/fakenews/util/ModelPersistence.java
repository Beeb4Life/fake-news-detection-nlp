package com.fakenews.util;

import java.io.*;

/**
 * Handles serializing and deserializing the trained NLP & ML pipeline to/from disk.
 */
public class ModelPersistence {

    public static void save(TrainedPipeline pipeline, File file) throws IOException {
        File parent = file.getParentFile();
        if (parent != null && !parent.exists()) {
            parent.mkdirs();
        }
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(file)))) {
            oos.writeObject(pipeline);
        }
    }

    public static TrainedPipeline load(File file) throws IOException, ClassNotFoundException {
        if (!file.exists()) {
            throw new FileNotFoundException("Model file not found: " + file.getAbsolutePath());
        }
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(file)))) {
            return (TrainedPipeline) ois.readObject();
        }
    }
}
