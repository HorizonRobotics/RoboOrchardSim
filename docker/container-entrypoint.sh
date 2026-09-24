#!/usr/bin/env bash
set -euo pipefail

current_uid="$(id -u)"
current_gid="$(id -g)"

# Some runtimes start containers with a host UID/GID that is unknown to the
# image. Use an in-memory NSS database so the image also works when it cannot
# modify /etc/passwd (for example, when the runtime starts as non-root).
if ! getent passwd "${current_uid}" >/dev/null \
    || ! getent group "${current_gid}" >/dev/null; then
    nss_dir="$(mktemp -d "${TMPDIR:-/tmp}/nss-wrapper-${current_uid}.XXXXXX")"
    passwd_file="${nss_dir}/passwd"
    group_file="${nss_dir}/group"

    cp /etc/passwd "${passwd_file}"
    cp /etc/group "${group_file}"

    if ! getent group "${current_gid}" >/dev/null; then
        runtime_group="container-${current_gid}"
        while getent group "${runtime_group}" >/dev/null; do
            runtime_group="_${runtime_group}"
        done
        printf '%s:x:%s:\n' \
            "${runtime_group}" "${current_gid}" >> "${group_file}"
    fi

    if ! getent passwd "${current_uid}" >/dev/null; then
        runtime_user="${CONTAINER_USER_NAME:-container-${current_uid}}"
        runtime_user="${runtime_user//[^a-zA-Z0-9_.-]/_}"
        if [[ -z "${runtime_user}" ]]; then
            runtime_user="container-${current_uid}"
        fi
        while getent passwd "${runtime_user}" >/dev/null; do
            runtime_user="_${runtime_user}"
        done

        runtime_home="${CONTAINER_HOME:-${nss_dir}/home}"
        mkdir -p "${runtime_home}"

        printf '%s:x:%s:%s:Container user:%s:/bin/bash\n' \
            "${runtime_user}" \
            "${current_uid}" \
            "${current_gid}" \
            "${runtime_home}" >> "${passwd_file}"
    fi

    nss_library="/usr/lib/x86_64-linux-gnu/libnss_wrapper.so"
    if [[ ! -f "${nss_library}" ]]; then
        echo "container-entrypoint: ${nss_library} not found" >&2
        exit 1
    fi

    export NSS_WRAPPER_PASSWD="${passwd_file}"
    export NSS_WRAPPER_GROUP="${group_file}"
    export LD_PRELOAD="${nss_library}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

passwd_record="$(getent passwd "${current_uid}")"
export USER="$(cut -d: -f1 <<< "${passwd_record}")"
export LOGNAME="${USER}"
export HOME="$(cut -d: -f6 <<< "${passwd_record}")"

# Preserve the old image's `docker run IMAGE -lc ...` interface while also
# allowing an executable to be passed directly.
if [[ $# -eq 0 ]]; then
    set -- /bin/bash
elif [[ "${1}" == -* ]]; then
    set -- /bin/bash "$@"
fi

exec "$@"
