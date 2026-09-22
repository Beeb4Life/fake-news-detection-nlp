#!/usr/bin/env bash
set -e

echo "Compiling and packaging Fake News Detection NLP application..."
mvn clean package -DskipTests
echo "Build complete! You can run the executable JAR with:"
echo "  java -jar target/fake-news-detector-1.0.0.jar"
