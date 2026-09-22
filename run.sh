#!/usr/bin/env bash
set -e

# Run the Fake News Detection application via Maven
mvn compile exec:java -Dexec.mainClass="com.fakenews.Main" -q
