def get_bit_i32 i x =
  let b = i32.get_bit i x
  in if i == i32.num_bits - 1 then b ^ 1 else b

local
def old_radix_sort_step [n] 't
                        (xs: [n]t)
                        (get_bit: i32 -> t -> i32)
                        (digit_n: i32) : [n]t =
  let num x = get_bit (digit_n + 1) x * 2 + get_bit digit_n x
  let pairwise op (a1, b1, c1, d1) (a2, b2, c2, d2) = (a1 `op` a2, b1 `op` b2, c1 `op` c2, d1 `op` d2)
  let bins = xs |> map num
  let flags =
    bins
    |> map (\x ->
              ( i64.bool (x == 0)
              , i64.bool (x == 1)
              , i64.bool (x == 2)
              , i64.bool (x == 3)
              ))
  let offsets = scan (pairwise (+)) (0, 0, 0, 0) flags
  let (na, nb, nc, _nd) = last offsets
  let f bin (a, b, c, d) =
    (-1)
    + a * (i64.bool (bin == 0))
    + na * (i64.bool (bin > 0))
    + b * (i64.bool (bin == 1))
    + nb * (i64.bool (bin > 1))
    + c * (i64.bool (bin == 2))
    + nc * (i64.bool (bin > 2))
    + d * (i64.bool (bin == 3))
  let is = map2 f bins offsets
  in scatter (copy xs) is xs

-- | Old Radix Sort
def old_radix_sort [n] 't
                   (num_bits: i32)
                   (get_bit: i32 -> t -> i32)
                   (xs: [n]t) : [n]t =
  let iters = if n == 0 then 0 else (num_bits + 2 - 1) / 2
  in loop xs for i < iters do old_radix_sort_step xs get_bit (i * 2)

local
def old_with_fuse_radix_sort_step [n] 't
                                  (xs: [n]t)
                                  (get_bit: i32 -> t -> i32)
                                  (digit_n: i32) : [n]t =
  let num x = i8.i32 <| get_bit (digit_n + 1) x * 2 + get_bit digit_n x
  let pairwise op (a1, b1, c1, d1) (a2, b2, c2, d2) = (a1 `op` a2, b1 `op` b2, c1 `op` c2, d1 `op` d2)
  let pairwise' op (a1, b1, c1) (a2, b2, c2) = (a1 `op` a2, b1 `op` b2, c1 `op` c2)
  let bins = map num xs
  let flags =
    bins
    |> map (\x -> (x == 0, x == 1, x == 2, x == 3))
  let flags' =
    map num xs
    |> map (\x ->
              ( i64.bool (x == 0)
              , i64.bool (x == 1)
              , i64.bool (x == 2)
              ))
  let offsets =
    flags
    |> map (\(a, b, c, d) -> (i64.bool a, i64.bool b, i64.bool c, i64.bool d))
    |> scan (pairwise (+)) (0, 0, 0, 0)
  let (na, nb, nc) = reduce_comm (pairwise' (+)) (0, 0, 0) flags'
  let f bin (a, b, c, d) =
    (-1)
    + a * (i64.bool (bin == 0))
    + na * (i64.bool (bin > 0))
    + b * (i64.bool (bin == 1))
    + nb * (i64.bool (bin > 1))
    + c * (i64.bool (bin == 2))
    + nc * (i64.bool (bin > 2))
    + d * (i64.bool (bin == 3))
  let is = map2 f bins offsets
  in scatter (#[scratch] copy xs) is xs

-- | Old radix sort with reduce.
def old_with_fuse_radix_sort [n] 't
                             (num_bits: i32)
                             (get_bit: i32 -> t -> i32)
                             (xs: [n]t) : [n]t =
  let iters = if n == 0 then 0 else (num_bits + 2 - 1) / 2
  in loop xs for i < iters do old_with_fuse_radix_sort_step xs get_bit (i * 2)

local
def exscan xs =
  map2 (-) (scan (+) 0i64 xs) xs

local
def get_bin 't
            (k: i32)
            (get_bit: i32 -> t -> i32)
            (digit_n: i32)
            (x: t) : i64 =
  i64.i32
  <| loop acc = 0
     for i < k do
       acc + (get_bit (digit_n + i) x << i)

local
def radix_sort_step_i16 [n] 't
                        (get_bit: i32 -> t -> i32)
                        (digit_n: i32)
                        (xs: [n]t) : ([n]t, [4]i64, [4]i16) =
  let num x = i8.i32 <| get_bit (digit_n + 1) x * 2 + get_bit digit_n x
  let pairwise op (a1, b1, c1, d1) (a2, b2, c2, d2) = (a1 `op` a2, b1 `op` b2, c1 `op` c2, d1 `op` d2)
  let bins = map num xs
  let flags =
    bins
    |> map (\x -> (x == 0, x == 1, x == 2, x == 3))
  let offsets =
    flags
    |> map (\(a, b, c, d) -> (i16.bool a, i16.bool b, i16.bool c, i16.bool d))
    |> scan (pairwise (+)) (0, 0, 0, 0)
  let counts = hist (+) 0 4 (map (i64.i8 <-< num) xs) (rep 1)
  let (na, nb, nc, nd) = (counts[0], counts[1], counts[2], counts[3])
  let f bin (a, b, c, d) =
    i64.i16 ((-1)
             + a * (i16.bool (bin == 0))
             + na * (i16.bool (bin > 0))
             + b * (i16.bool (bin == 1))
             + nb * (i16.bool (bin > 1))
             + c * (i16.bool (bin == 2))
             + nc * (i16.bool (bin > 2))
             + d * (i16.bool (bin == 3)))
  let is = map2 f bins offsets
  in ( scatter (#[scratch] copy xs) is xs
     , [i64.i16 na, i64.i16 nb, i64.i16 nc, i64.i16 nd]
     , [0, na, na + nb, na + nb + nc]
     )

local
def blocked_radix_sort_step [n] [m] [r] 't
                            (get_bit: i32 -> t -> i32)
                            (digit_n: i32)
                            (xs: *[n * m + r]t) =
  let (blocks, rest) = split xs
  let (sorted_rest, hist_rest, offsets_rest) =
    radix_sort_step_i16 get_bit digit_n rest
  let (sorted_blocks, hist_blocks, offsets_blocks) =
    unflatten blocks
    |> map (radix_sort_step_i16 get_bit digit_n)
    |> unzip3
  let histograms = hist_blocks ++ [hist_rest]
  let sorted = sized (n * m + r) (flatten sorted_blocks ++ sorted_rest)
  let old_offsets = offsets_blocks ++ [offsets_rest]
  let new_offsets =
    histograms
    |> transpose
    |> flatten
    |> exscan
    |> unflatten
    |> transpose
  let is =
    map2 (\i elem ->
            let bin = get_bin 2 get_bit digit_n elem
            let block_idx = i / m
            let new_offset = new_offsets[block_idx][bin]
            let old_block_offset = i64.i16 old_offsets[block_idx][bin]
            let old_offset = m * block_idx + old_block_offset
            let idx = (i - old_offset) + new_offset
            in idx) (iota (n * m + r)) sorted
  in scatter (#[scratch] copy xs) is sorted

local
def (///) (a: i32) (b: i32) : i32 =
  a / b + i32.bool (a % b != 0)

local
def (////) (a: i64) (b: i64) : i64 =
  a / b + i64.bool (a % b != 0)

def blocked_radix_sort [n] 't
                       (block: i16)
                       (num_bits: i32)
                       (get_bit: i32 -> t -> i32)
                       (xs: [n]t) : [n]t =
  let iters = if n == 0 then 0 else (num_bits + 2 - 1) / 2
  let block = i64.i16 block
  let n_blocks = n / block
  let rest = n % block
  let xs = sized (n_blocks * block + rest) xs
  in sized n
     <| loop xs = copy xs
        for i < iters do
          blocked_radix_sort_step get_bit (i * 2) xs

-- ==
-- entry: expected
-- input @ ../data/randomints_moderate_500MiB.in
entry expected = old_radix_sort i32.num_bits get_bit_i32

-- ==
-- input @ ../data/randomints_moderate_500MiB.in
-- output @ randomints_moderate_500MiB.out
entry main = old_with_fuse_radix_sort i32.num_bits get_bit_i32

-- ==
-- entry: blocked
-- input @ ../data/randomints_moderate_500MiB.in
-- output @ randomints_moderate_500MiB.out
entry blocked = blocked_radix_sort 256 i32.num_bits get_bit_i32
