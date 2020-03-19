#!/usr/local_rwth/bin/zsh

# for target in intel clang chameleon
for target in intel chameleon_manual
do
  # load default modules
  module purge
  module load DEVELOP

  # load target specific compiler and libraries
  while IFS='' read -r line || [[ -n "$line" ]]; do
    if  [[ $line == LOAD_COMPILER* ]] || [[ $line == LOAD_LIBS* ]] ; then
      eval "$line"
    fi
  done < "flags_${target}.def"
  module load ${LOAD_COMPILER}
  module load intelmpi/2018
  module load ${LOAD_LIBS}
  module li

  # make corresponding targets
  TARGET=${target} make -C pure-parallel clean all
  #TARGET=${target} make -C singlecom-deps clean all
  #TARGET=${target} make -C fine-deps clean all
done
