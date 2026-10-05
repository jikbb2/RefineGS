#!/usr/bin/env bash
# Detect -- and if asked, prevent -- any change a baseline makes to our data.
#
# Why: Split&Splat's own scripts rewrite intermediates in place and rename image
# extensions. Running a baseline against the same tree the main pipeline reads is how a
# result silently changes meaning between two runs. Isolation by path is the first defence;
# this is the second, so that "I think it touched something" becomes "these three files
# changed, here they are".
#
#   bash bench_guard.sh snap  ~/RefineGS/data/replica_room0_v2  /tmp/room0.snap
#   ... run the baseline ...
#   bash bench_guard.sh check ~/RefineGS/data/replica_room0_v2  /tmp/room0.snap
#
#   bash bench_guard.sh freeze ~/RefineGS/data/replica_room0_v2   # read-only, hard stop
#   bash bench_guard.sh thaw   ~/RefineGS/data/replica_room0_v2   # undo
#
# check exits 1 when anything differs, so it can gate a script.
set -uo pipefail

MODE=${1:-help}
DIR=${2:-}
SNAP=${3:-}

# -P (the default) does not follow symlinks, so a snapshot records the LINK, not the file it
# points at: images/ is a tree of symlinks into nice-slam and we want to know if the links
# themselves were renamed or replaced, which is exactly the failure being guarded against.
list_tree() {
  find "$1" \( -type f -o -type l -o -type d \) \
       -printf '%y\t%s\t%T@\t%l\t%P\n' 2>/dev/null \
    | sed 's/\.[0-9]*\t/\t/' | LC_ALL=C sort -t $'\t' -k5,5
}

case "${MODE}" in
  snap)
    [ -d "${DIR}" ] || { echo "[abort] not a directory: ${DIR}"; exit 1; }
    [ -n "${SNAP}" ] || { echo "[abort] give a snapshot path"; exit 1; }
    list_tree "${DIR}" > "${SNAP}"
    echo "[snap] $(wc -l < "${SNAP}") entries  ${DIR} -> ${SNAP}"
    ;;

  check)
    [ -d "${DIR}" ] || { echo "[abort] not a directory: ${DIR}"; exit 1; }
    [ -f "${SNAP}" ] || { echo "[abort] no snapshot at ${SNAP}"; exit 1; }
    now=$(mktemp); list_tree "${DIR}" > "${now}"
    # Compare on the path column so a renamed file shows up as one gone and one new, which
    # is what an extension rewrite looks like and the single thing most worth catching.
    old_p=$(mktemp); new_p=$(mktemp)
    cut -f5 "${SNAP}" | LC_ALL=C sort > "${old_p}"
    cut -f5 "${now}"  | LC_ALL=C sort > "${new_p}"
    gone=$(LC_ALL=C comm -23 "${old_p}" "${new_p}")
    made=$(LC_ALL=C comm -13 "${old_p}" "${new_p}")
    # Directories are kept in the path set so a deleted folder is caught, but their mtime is
    # skipped: a directory's time changes whenever a file inside it does, which the NEW and
    # REMOVED lines already say. Reporting it too buries the real finding.
    chg=$(LC_ALL=C join -t $'\t' -j 1 \
            <(awk -F'\t' '$1!="d"{print $5"\t"$1"\t"$2"\t"$3"\t"$4}' "${SNAP}" | LC_ALL=C sort -t $'\t' -k1,1) \
            <(awk -F'\t' '$1!="d"{print $5"\t"$1"\t"$2"\t"$3"\t"$4}' "${now}"  | LC_ALL=C sort -t $'\t' -k1,1) \
          | awk -F'\t' '$2!=$6 || $3!=$7 || $4!=$8 || $5!=$9 {print $1}')
    n_g=$(printf '%s\n' "${gone}" | grep -c . || true)
    n_m=$(printf '%s\n' "${made}" | grep -c . || true)
    n_c=$(printf '%s\n' "${chg}"  | grep -c . || true)
    echo "[check] ${DIR}"
    echo "        removed/renamed-from ${n_g}   new/renamed-to ${n_m}   modified ${n_c}"
    for lbl in gone made chg; do
      v=$(eval "printf '%s' \"\${${lbl}}\"")
      [ -n "${v}" ] || continue
      case "${lbl}" in gone) t="REMOVED ";; made) t="NEW     ";; chg) t="MODIFIED";; esac
      printf '%s\n' "${v}" | head -40 | sed "s/^/  ${t}  /"
      c=$(printf '%s\n' "${v}" | grep -c . || true)
      [ "${c}" -gt 40 ] && echo "  ... and $((c - 40)) more ${t}"
    done
    rm -f "${now}" "${old_p}" "${new_p}"
    if [ "${n_g}" -eq 0 ] && [ "${n_m}" -eq 0 ] && [ "${n_c}" -eq 0 ]; then
      echo "        clean -- nothing was touched"; exit 0
    fi
    exit 1
    ;;

  freeze)
    # The hard guarantee: the OS refuses the write instead of us noticing afterwards.
    # This also blocks OUR pipeline from writing there, so thaw before the next run.
    [ -d "${DIR}" ] || { echo "[abort] not a directory: ${DIR}"; exit 1; }
    chmod -R a-w "${DIR}" && echo "[freeze] ${DIR} is read-only -- thaw before the next RefineGS run"
    ;;

  thaw)
    [ -d "${DIR}" ] || { echo "[abort] not a directory: ${DIR}"; exit 1; }
    chmod -R u+w "${DIR}" && echo "[thaw] ${DIR} is writable again"
    ;;

  *)
    echo "Usage: bash bench_guard.sh snap|check|freeze|thaw DIR [SNAPSHOT]"
    echo "  snap   DIR FILE   record every path, size, mtime and symlink target"
    echo "  check  DIR FILE   diff against a snapshot; exit 1 if anything differs"
    echo "  freeze DIR        chmod -R a-w (blocks our own writes too)"
    echo "  thaw   DIR        undo freeze"
    ;;
esac
